import torch 
import numpy as np 
import librosa

# import math
import utils
import models
from utils.create_database import Speech2Text


def score_ctc(
        asr_config, 
        asr_model, 
        audio_file, 
        matrix_learned_token, 
        matrix_w):
    
    speech2text_kwargs = dict(
    asr_train_config=asr_config,
    asr_model_file=asr_model,
    )

    espnet_model = Speech2Text(**speech2text_kwargs)
    token_list = espnet_model.token_list
    # calculate score
    
    logprobs = []
    audio_, _ = librosa.load(audio_file)
        
    logits, results = espnet_model(audio_)
    logits = logits.log_softmax(-1)

    for n, (text, token, token_int, hyp) in zip(
            range(1, espnet_model.nbest + 1), results
        ):
        predicted_token = token        
        
    tensor_probs = np.exp(logits.numpy())

    for token in matrix_learned_token:
        probs = models.CTC.CTCforward(learned_token=token, matrix=tensor_probs, vocabs=token_list)
        logprobs.append(np.log(probs))

    score = sum([i * j for i, j in zip(logprobs, matrix_w)])
    return score, predicted_token