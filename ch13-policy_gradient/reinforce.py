""" Sutton and Barto 2nd edition, Chapter 13. Policy Gradient Methods
Implementation of REINFORCE algorithm for the short corridor example

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
def one_step(s,a):
    """receives current state and action s,a, returns reward and next state, r,
s_prime. a is either 0 (left) or 1 (right)
    """
    
    R = -1
    if s == 0:
        s_prime = a # left (0) goes to state 0, right (1) goes to state 1
    elif s == 1:
        s_prime = 2 if a ==0 else 0 # reversed motion
    elif s == 2:
        s_prime = 3 if a == 1 else 1
        
    return R, s_prime
        

################################################################################
def x(s,a):
    xs = np.asarray([[0,1],[1,0]])
    x = xs[a]
    return x


################################################################################
def compute_pi(theta, s):
    # compute soft-max for linear feature theta^T.x
    h = np.zeros(2)

    for a in range(2):
        h[a] = np.dot(x(s,a), theta)

    h -= np.max(h)

    pi = np.exp(h)/np.sum(np.exp(h))

    return pi


################################################################################
def compute_grad(theta,s,a):
    # compute soft-max for linear feature theta^T.x
    pi = compute_pi(theta,s)

    
    not_a = np.abs(a-1)
    
    grad = x(s,a) - pi[not_a] * x(s,not_a)


    return grad

def v(w,s):
    """ See Sutton and Barto 2nd edition 13.4, page 273 """
    return w


################################################################################
def REINFORCE(theta, gamma = 1.0, alpha=2**-13):
    EPISODES= 200

    GOAL = 3


    G_0s = []
    for ep in range(EPISODES):
        G = 0
        s = 0
        a = np.argmax(compute_pi(theta,s))

        ep_s = []
        ep_a = []
        ep_R = []

        ep_s.append(s)
        ep_a.append(a)

        R, s = one_step(s,a)
        
        ep_R.append(R)
        

        while s != GOAL:

            pi = compute_pi(theta, s)
            a = np.random.choice(2, p=pi)

            # print s,pi,a
            # if np.random.rand()<0.1:
            #     a = np.random.choice([0,1])
            

            R, s = one_step(s,a)

            ep_s.append(s)
            ep_a.append(a)
        
            ep_R.append(R)

        # print 'GOAL!'
        ep_R = np.asarray(ep_R)

        for t in range(len(ep_s)):
           G_t = np.sum(ep_R[t:])
           theta += alpha * (gamma**t) * G_t * compute_grad(theta, ep_s[t],ep_a[t])
           
        G_0 = np.sum(ep_R)

        #print 'ep %i, G_0 %f' % (ep, G_0), theta
        G_0s.append(G_0)
        
    return theta, np.asarray(G_0s)

 
################################################################################
# MAIN LOOP

RUNS = 100

theta = np.random.randn(2)

theta, G_0s = REINFORCE(theta)

G_0s = G_0s.reshape(1,-1)

for i in range(RUNS):
    theta = np.random.randn(2)
    theta, G_0s_t = REINFORCE(theta)
    G_0s = np.concatenate((G_0s, G_0s_t.reshape(1,-1)), axis=0)

    print 'RUN %i/%i' % (i,RUNS)

print G_0s.shape
print np.mean(G_0s,axis=0)


savefname = 'G.png'
plt.plot(np.mean(G_0s, axis=0))
plt.savefig(savefname)


os.system('open %s' %  savefname)
