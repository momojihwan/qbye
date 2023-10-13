import models
import utils
import numpy as np
import math

def threshold_ctc(matrix_learned_token, matrix_w, matrix_probs, vocabs):
    scores = []
    print(matrix_learned_token)

    for token in matrix_learned_token:
        logprobs = []
        for pr in matrix_probs:
            probs = models.CTC.CTCforward(learned_token=token, matrix=pr, vocabs=vocabs)
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