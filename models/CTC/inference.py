
from tqdm import tqdm 
import numpy as np 
import math 
import time 

import utils
import models

def inference_ctc(matrix_learned_phoneme, matrix_w, matrix_probs):
    model_path = "./models/KWS/"
    asr_model = 

    # chose threshold
    upper_boundary, lower_boundary = models.CTC.threshold_ctc(matrix_learned_phoneme, matrix_w, matrix_probs)
    print(upper_boundary, lower_boundary)
    threshold = lower_boundary
    while(1):
        input("press enter to record : ")
        utils.record()
        path = "record/example.wav"
        files = [path]
        score = models.CTC.score_ctc(path, matrix_learned_phoneme, matrix_w)

        for fname, trans in zip(files, asr_model.transcribe(paths2audio_files=files, logprobs=0)):
            phoneme = trans

        print(phoneme, "\n")
        print("threshold : ", threshold)
        print("score : ", score)
        if score > threshold:
            print("KEYWORD")
        else:
            print("NON-KEYWORD")
            