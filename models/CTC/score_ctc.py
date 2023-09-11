import torch 
import numpy as np 
import torch.tensor
# import math
import util
import models


def score_stream(data, matrix_learned_phoneme, matrix_w):
    model_path = "./checkpoints/CTC/"
    asr_model = 
    logprobs = []
    tensor_probs = asr_model.transcribe(data, logprobs=1)
    tensor_probs = np.exp(tensor_probs.numpy())

    for phoneme in matrix_learned_phoneme:
        probs = models.CTC.CTCforward(learned_phoneme=phoneme, matrix=tensor_probs)
        logprobs.append(np.log(probs))
        
    score = sum([i*j for i, j in zip(logprobs, matrix_w)])
    return score

def score_ctc(path, matrix_learned_phoneme, matrix_w):
    model_path = "./checkpoints/CTC/"
    asr_model = 
    # calculate score
    files = [path]
    logprobs = []
    for fname, prob in zip(files, asr_model.transcribe(paths2audio_files=files, logprobs=1)):
        tensor_probs = prob
    
    tensor_probs = np.exp(tensor_probs.numpy())

    for phoneme in matrix_learned_phoneme:
        probs = models.CTC.CTCforward(learned_phoneme=phoneme, matrix=tensor_probs)
        logprobs.append(np.log(probs))

    score = sum([i * j for i, j in zip(logprobs, matrix_w)])
    return score