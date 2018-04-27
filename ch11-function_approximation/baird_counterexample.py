""" Sutton and Barto 2nd edition, Exercise 11.3
Divergence of semi-gradient Q-learning in Baird's counterexample

jlezama@fing.edu.uy
"""

import os
import numpy as np
from scipy.stats.distributions import poisson


import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt



################################################################################


def one_step(w_t, x_t, x_tp1, rho_t, gamma=0.99, alpha=0.01, R_tp1=0):
    
    delta_t = R_tp1 + gamma * np.dot(w_t,x_tp1) - np.dot(w_t, x_t)
    w_tp1 = w_t + alpha * rho_t * delta_t * x_t
    return w_tp1


################################################################################
Ns = 7 # number of  states

# states matrix
X = np.asarray([[2,0,0,0,0,0,0,1],[0,2,0,0,0,0,0,1],[0,0,2,0,0,0,0,1],[0,0,0,2,0,0,0,1],[0,0,0,0,2,0,0,1],[0,0,0,0,0,2,0,1],[0,0,0,0,0,0,1,2]])



# initial state is state 7
i_t = Ns
x_t = X[i_t-1]

# initital weight vector
w_t = np.ones(8)



for it in range(100000):
    if i_t == Ns:
        i_tp1 = np.random.randint(Ns+1)
    else:
        i_tp1 = Ns

    # importance sampling: target policy over behavior
    if i_t ==Ns and i_tp1 <Ns:
        rho_t = 0/(7/6.) # never seen in pi
    elif i_t == Ns and i_tp1 == Ns:
        rho_t = 7/1.
    else:
        rho_t = 1

    x_tp1 =  X[i_tp1-1]
    w_t = one_step(w_t, x_t, x_tp1, rho_t)

    i_t = i_tp1
    x_t = x_tp1


print w_t, np.dot(x_t,w_t)
