
from tqdm import tqdm 
import logging

import numpy as np
import os
import models
from utils.create_database import create_database_ctc

def make_example_list(audio_path):
    audio_file = []
    audio_file_list = os.listdir(audio_path)
    for file in audio_file_list:
        audio_file.append(os.path.join(audio_path,file))
    return audio_file

# def inference_ctc(matrix_learned_phoneme, matrix_w, matrix_probs):
def inference_ctc(asr_config, model_path, audio_path):
    audio_file = make_example_list(audio_path)
    matrix_learned_token, matrix_w, matrix_probs, token_list = create_database_ctc(asr_config, model_path, audio_file)
    
    # chose threshold
    upper_boundary, lower_boundary = models.CTC.threshold_ctc(matrix_learned_token, matrix_w, matrix_probs, token_list)
    threshold = lower_boundary
    # while(1):
        # input("press enter to record : ")
        # utils.record()
    path = "record/example.wav"
    files = [path]
    score, predicted_token = models.CTC.score_ctc(asr_config, model_path, path, matrix_learned_token, matrix_w)
    print(predicted_token, "\n")
    print("threshold : ", threshold)
    print("score : ", score)
    if score > threshold:
        print("KEYWORD")
    else:
        print("NON-KEYWORD")
            