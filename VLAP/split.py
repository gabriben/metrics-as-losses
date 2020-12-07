import numpy as np
import pandas as pd
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

def split(X, y, s, r):

    rs = RandomState(MT19937(SeedSequence(r)))
    
    i = X.index.to_numpy()
    np.random.shuffle(i, )
    P = np.rint(np.multiply(i.size, s))
    P[-1] = len(i) - np.sum(P[:-1])
    P = np.cumsum(P)
    P = P - 1
    P = [int(p) for p in P]

    last = -1
    X_splitted = []
    for j, p in enumerate(P): 
        X_splitted.append(X[(last + 1):p])
        last = p

    last = -1
    y_splitted = []
    for j, p in enumerate(P): 
        y_splitted.append(y[(last + 1):p])
        last = p

    return(X_splitted + y_splitted)


# X = pd.DataFrame({'col1' : np.arange(103), 'col2':np.arange(103)})
# y = pd.DataFrame({'col1' : np.arange(103), 'col2':np.arange(103)})

# a = split(X, y, [0.8, 0.2] , 44)

# example
# X = pd.DataFrame({'col1' : np.arange(103), 'col2':np.arange(103)})
# # X = np.arange(103)
# i = X.index.to_numpy()
# s = [0.2, 0.2, 0.6]
# np.random.shuffle(i)
# P = np.rint(np.multiply(i.size, s))
# P[-1] = len(i) - np.sum(P[:-1])
# P = np.cumsum(P)
# P = P - 1
# P = [int(p) for p in P]

# last = -1
# X_splitted = []
# for j, p in enumerate(P): 
#     X_splitted.append(X[(last + 1):p])
#     last = p
