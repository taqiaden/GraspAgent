import numpy as np

selection_prob_k=0.1

def get_selection_probabilty(indexes):
    indexes=np.asarray(indexes,dtype=int)
    sort_index=np.argsort(indexes)
    if selection_prob_k is None:
        k=np.random.randint(1,10)
    else:
        k=selection_prob_k

    selection_p=((1+sort_index)/np.max(sort_index))**k
    return selection_p
