import numpy as np

# create CTC forward function
def CTCforward(learned_phoneme, matrix):
    # matrix of inference file (probs version, not log version)
    vocabs = [' ', 'A']

    phoneme_list = learned_phoneme.split(" ")
    l = []
    for i in phoneme_list:
        l.append("-")
        l.append(i)
        l.append("-")
        l.append(" ")

    l.pop()

    phoneme_list = list(set(phoneme_list))
    phoneme_list.extend([" ", "-"])
    idx_col = []
    for ele in l:
        idx_col.append(vocabs.index(ele))
    
    matrix = np.array(matrix).T

    matrix_l = np.array([matrix[i] for i in idx_col])
    
    for i in range(1, matrix_l.shape[1]):
        for j in range(matrix_l.shape[0]):
            p = matrix_l[j, i]
            if l[j] == '-' or (j>1 and l[j]==l[j=2]):
                matrix_l[j, i] = (matrix_l[j, i-1] + matrix_l[j-1, i-1]) * p
            else:
                matrix_l[j, i] = (matrix_l[j, i-1] + matrix_l[j-1, i-1] + matrix_l[j-2, i-1]) * p

    return matrix_l[-2, -1] + matrix_l[-1, -1]
