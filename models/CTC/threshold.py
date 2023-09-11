import models
import util
import numpy as np
import math

def threshold_ctc(matrix_learned_phoneme, matrix_w, matrix_probs):
    scores = []
    print(matrix_learned_phoneme)

    for phoneme in matrix_learned_phoneme:
        logprobs = []
        for pr in matrix_probs:
            probs = models.CTC.CTCforward(learned_phoneme=phoneme, matrix=pr)
            logprobs.append(np.log(probs))

        score = sum([logprobs[i] * matrix_w[i] for i in range(len(matrix_w))])
        scores.append(score)

    mean = np.mean(scores)

    std = np.std(scores)
    interval = 1.96 * std / math.sqrt(len(scores))
    print("scores: ", scores)
    print("mean: ", mean)
    print("std: ", std)
    print("interval: ", interval)

    upper_boundary = mean + interval
    lower_boundary = mean - 2 * interval
    print("upper_ : ", upper_boundary)
    print("lower_ : ", lower_boundary)
    return upper_boundary, lower_boundary