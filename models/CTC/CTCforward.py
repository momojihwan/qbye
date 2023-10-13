import numpy as np

# create CTC forward function
def CTCforward(learned_token, matrix, vocabs):
    '''
    learned_token : predicted tokens ['_CLO', 'S']
    matrix : logits's log_softmax[L, V]
    vocabs : token list [V] 
    '''
    # matrix of inference file (probs version, not log version)
    # pred_token_list = learned_token.split(" ")
    
    l = []
    for i in learned_token:
        l.append("<blank>")
        l.append(i)
        l.append("<blank>")
    # l = ['<blank>', '_CLO', '<blank>', '<blank>', 'S']
        
    l.pop()
    token_list = list(set(learned_token))
    
    token_list.extend(["<blank>"])
    idx_col = []
    for ele in l:
        idx_col.append(vocabs.index(ele))
    # idx_col = [0, 884, 0, 0, 3]
    
    matrix = np.array(matrix).T

    matrix_l = np.array([matrix[i] for i in idx_col])
    
    for i in range(1, matrix_l.shape[1]):
        for j in range(matrix_l.shape[0]):
            p = matrix_l[j, i]
            if l[j] == '<blank>' or (j>1 and l[j]==l[j-2]):
                matrix_l[j, i] = (matrix_l[j, i-1] + matrix_l[j-1, i-1]) * p
            else:
                matrix_l[j, i] = (matrix_l[j, i-1] + matrix_l[j-1, i-1] + matrix_l[j-2, i-1]) * p

    return matrix_l[-2, -1] + matrix_l[-1, -1]
