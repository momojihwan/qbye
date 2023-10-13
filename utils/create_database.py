import torch 
import torch.quantization
import numpy as np 
import models
import numpy as np 
import math
import librosa

import logging
import math 
import time 

import sys
sys.path.append('~/Workspace/espnet')

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
from typeguard import check_argument_types, check_return_type

from espnet2.torch_utils.device_funcs import to_device
from espnet2.tasks.asr import ASRTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch, Hypothesis

try:
    from transformers import AutoModelForSeq2SeqLM
    from transformers.file_utils import ModelOutput

    is_transformers_available = True
except ImportError:
    is_transformers_available = False

class Speech2Text:
    """Speech2Text class for Transducer models.

    Args:
        asr_train_config: ASR model training config path.
        asr_model_file: ASR model path.
    """

    def __init__(
        self,
        asr_train_config: Union[Path, str] = None,
        asr_model_file: Union[Path, str] = None,
    ) -> None:
        """Construct a Speech2Text object."""
        super().__init__()

        assert check_argument_types()

        token_type = None
        bpemodel = None
        device = "cpu"
        beam_size = 5
        dtype = "float32"
        lm_weight = 0.0
        nbest = 1
        ctc_weight = 1.0
        ngram_weight = 0.0
        penalty = 0.0

        task = ASRTask

        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = task.build_model_from_file(
            asr_train_config, asr_model_file, device
        )

        asr_model.to(dtype=getattr(torch, dtype)).eval()

        decoder = asr_model.decoder

        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        scorers["lm"] = None
        scorers["ngram"] = None

        hugging_face_model = None
        hugging_face_linear_in = None

        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )

        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )

        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif (
            token_type == "bpe"
            or token_type == "hugging_face"
            or "whisper" in token_type
        ):
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)

        if bpemodel not in ["whisper_en", "whisper_multilingual"]:
            converter = TokenIDConverter(token_list=token_list)
        else:
            converter = OpenAIWhisperTokenIDConverter(model_type=bpemodel)
            beam_search.set_hyp_primer(
                list(converter.tokenizer.sot_sequence_including_notimestamps)
            )
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.token_list = token_list
        self.converter = converter
        self.tokenizer = tokenizer
        self.maxlenratio = 0.0
        self.minlenratio = 0.0
        self.beam_search = beam_search
        self.hugging_face_model = hugging_face_model
        self.hugging_face_linear_in = hugging_face_linear_in
        self.hugging_face_beam_size = beam_size
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ):
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        logging.info("speech length: " + str(speech.size(1)))
        
        enc, _ = self.asr_model.encode(**batch)

        logits = self.asr_model.ctc.ctc_lo(enc)
        logits = logits[0]

        results = self._decode_single_sample(enc[0])


        return logits, results


    def _decode_single_sample(self, enc: torch.Tensor):

        if self.hugging_face_model:
            decoder_start_token_id = (
                self.hugging_face_model.config.decoder_start_token_id
            )
            yseq = self.hugging_face_model.generate(
                encoder_outputs=ModelOutput(
                    last_hidden_state=self.hugging_face_linear_in(enc).unsqueeze(0)
                ),
                use_cache=True,
                decoder_start_token_id=decoder_start_token_id,
                num_beams=self.hugging_face_beam_size,
                max_length=self.hugging_face_decoder_max_length,
            )
            nbest_hyps = [Hypothesis(yseq=yseq[0])]
            logging.info(
                "best hypo: "
                + "".join(self.converter.ids2tokens(nbest_hyps[0].yseq[1:]))
                + "\n"
            )
        else:
            nbest_hyps = self.beam_search(
                x=enc, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
            )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            # remove sos/eos and get results
            last_pos = None if self.asr_model.use_transducer_decoder else -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        return results



    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Speech2Text instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Text: Speech2Text instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Speech2Text(**kwargs)
    
def create_database_ctc(
        asr_config, 
        asr_model,
        audio_file,
        ):
    
    speech2text_kwargs = dict(
    asr_train_config=asr_config,
    asr_model_file=asr_model,
    )

    espnet_model = Speech2Text(**speech2text_kwargs)
    token_list = espnet_model.token_list
    matrix_w = []
    matrix_learned_token = []
    matrix_probs = []
    for file in audio_file:
        audio_, _ = librosa.load(file)
        
        logits, results = espnet_model(audio_)

        for n, (text, token, token_int, hyp) in zip(
                    range(1, espnet_model.nbest + 1), results
                ):
            logits = logits.log_softmax(-1)
            matrix_learned_token.append(token)
            learned_token_probs = models.CTC.CTCforward(token, np.exp(logits.numpy()), token_list)
            
            matrix_probs.append(np.exp(logits.numpy()))
            w = -1 / np.log(learned_token_probs)
            
            matrix_w.append(w)
        
        # matrix_learned_token.append(nbest_hyps)
        
        # matrix_probs.append(np.exp(logits.numpy()))

        # weight (confidence of learned_phoneme)


    return matrix_learned_token, matrix_w, matrix_probs, token_list