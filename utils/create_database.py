import torch 
import numpy as np 
import torch.tensor
import math
import utils
import models
import numpy as np 


def create_database_ctc(args):
    model_path = "./checkpoints/CTC/"
    asr_model = 
    matrix_w = []
    matrix_learned_phoneme = []
    matrix_probs = []
    for file in args:
        files = [file]

        # tensor of  the probs(log) each step
        for fname, probs in zip(files, asr_model.transcribe(paths2audio_files=files, logprobs=1)):
            tensor_probs = probs
        for fname, phoneme in zip(files, asr_model.transcribe(paths2audio_files=files, logprobs=0)):
            learned_phoneme = phoneme

        matrix_learned_phoneme.append(learned_phoneme)
        learned_phoneme_probs = models.CTC.CTCforward(learned_phoneme, np.exp(tensor_probs.numpy()))
        matrix_probs.append(np.exp(tensor_probs.numpy()))

        # weight (confidence of learned_phoneme)
        w = -1 / np.log(learned_phoneme_probs)
        matrix_w.append(w)

    return matrix_learned_phoneme, matrix_w, matrix_probs