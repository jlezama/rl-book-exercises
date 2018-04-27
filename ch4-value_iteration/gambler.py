""" Gambler Problem. Exercise 4.9 Sutton and Barto 2nd edition.
jlezama@fing.edu.uy
"""
import os
import numpy as np
from scipy.stats.distributions import poisson
import matplotlib
import matplotlib.pyplot as plt

def value_iteration(V, pi, PR, gamma):
    global THETA

    Delta = np.inf

    it = 0

    while Delta>THETA:
        print 'entering while'
        Delta = 0
        for s in range(100):
            v = V[s]

            argmax_a = -np.inf
            max_return = -np.inf
            for a in range(min(s,100-s)+1):
                expected_return = 0
                for s_prime in [s-a, s+a]:
                    P, R = PR(s,a,s_prime)
                    expected_return += P * (R + gamma * V[s_prime])

                    
                if expected_return> max_return:
                    max_return = expected_return
                    argmax_a = a
                # if expected_return == max_return:
                #     argmax_a = np.random.choice([a, argmax_a])

            V[s] = max_return
            pi[s] = argmax_a
            Delta = max(Delta, np.abs(v-V[s]))
        
        it+=1
        plot(V, pi, it)
            
    return V, pi

def PR(s, a, s_prime):
    global p_h


    # with probabilty p_h you get s+a, with probability 1-p_h you get s-a
    if s_prime == s+a:
        return p_h, int(s_prime==100)
    elif s_prime == s-a:
        return 1-p_h, 0

    else:
        return 0, 0


def plot(V, pi, it):
    global fig, axes
    os.system("mkdir -p gambler_figures")
    
    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    im = ax.plot(V)
    ax.set_title('V')
    
    ax = axes[1]
    im = ax.bar(range(101),pi)
    ax.set_title('pi')
    
    plt.savefig('gambler_figures/result_iter_%02i.png' % it)

    plt.clf()

################################################################################ 
# MAIN LOOP

THETA = 1e-16
p_h = 0.4

gamma = 1;

V = np.zeros(101)
pi = np.zeros(101)


V, pi = value_iteration(V, pi, PR, gamma)


