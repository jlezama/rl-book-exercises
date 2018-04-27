""" Windy Gridworld Problem. Exercises 6.9 and 6.10, Sutton and Barto 2nd edition.
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
# MAIN FUNCTIONS
def sarsa(Q, EPISODES, alpha, gamma,  eps, stochastic_wind):
    global wind, grid, START, GOAL

    H, W = wind.shape
    

    for episode in range(EPISODES):
    
        Is, Js = np.where(grid==START)
        # S = (Is, Js)
        
        A, vx, vy = epsilon_greedy(Q, Is, Js, eps)

        steps = 0
        
        while grid[Is, Js] != GOAL:
            # take action A
            Is_prime = Is + vy - wind[Is,Js] + stochastic_wind * ( wind[Is,Js]>0) * np.random.choice([-1,0,1])
            Js_prime = Js + vx
    
            Is_prime = min(H-1, max(0,Is_prime))
            Js_prime = min(W-1, max(0,Js_prime))
    
            # choose A_prime from S_prime
            A_prime, vx_prime, vy_prime = epsilon_greedy(Q, Is_prime, Js_prime, eps)
    
            Q[Is, Js, A] += alpha * (-1 + gamma*Q[Is_prime, Js_prime, A_prime] - Q[Is,Js,A])

            Is = Is_prime
            Js = Js_prime

            vx = vx_prime
            vy = vy_prime
            A = A_prime

            steps+=1
    
            if steps %1000 ==0:
                print 'step %i' % steps, Is, Js, Is_prime, Js_prime, vy, vx
               
        print 'reached goal in %i steps, episode %i'% (steps, episode)
    
    return Q

################################################################################
# AUX FUNCTIONS

def epsilon_greedy(Q, Is, Js, eps):
    global actions


    best_actions_ix = np.where(Q[Is, Js, actions]==np.max(Q[Is, Js, actions]))[0]


    if len(best_actions_ix)>1:
        best_action_ix = np.random.choice(best_actions_ix)
    else:
        best_action_ix = best_actions_ix[0]

    best_action = actions[best_action_ix] if np.random.rand()>eps else np.random.choice(actions)

    vx, vy = action_to_pair(best_action)

    return best_action, vx, vy

def action_to_pair(a):
    assert(a>=0 and a<9)
    vx = int(np.floor(a/3)-1)
    vy = int(np.mod(a,3)-1)
    return vx, vy

def pair_to_action(vx, vy):
    assert np.abs(vx)<=1 and np.abs(vy)<=1

    return int((vx+1)*3 + vy + 1)


# 0 (-1, -1)
# 1 (-1, 0)
# 2 (-1, 1)
# 3 (0, -1)
# 4 (0, 0)
# 5 (0, 1)
# 6 (1, -1)
# 7 (1, 0)
# 8 (1, 1)

# non-king moves: 1, 3, 5, 7
# king moves: 0, 1, 2, 3, 5, 6, 7, 8

def plot_sample(Q, it, stochastic_wind, eps=0):
    global wind

    plt.clf()
    x_t = []
    y_t = []

    H, W = wind.shape
    
    Is, Js = np.where(grid==START)
            
    A, vx, vy = epsilon_greedy(Q, Is, Js, eps)

    steps = 0

    y_t.append(Is)
    x_t.append(Js)
        
    while grid[Is, Js] != GOAL and steps <5000:
            # take action A
            Is_prime = Is + vy - wind[Is,Js] + stochastic_wind * ( wind[Is,Js]>0) * np.random.choice([-1,0,1])
            Js_prime = Js + vx
    
            Is_prime = min(H-1, max(0,Is_prime))
            Js_prime = min(W-1, max(0,Js_prime))
    
            # choose A_prime from S_prime
            A_prime, vx_prime, vy_prime = epsilon_greedy(Q, Is_prime, Js_prime, eps)
    
    
            Is = Is_prime
            Js = Js_prime

            vx = vx_prime
            vy = vy_prime
            A = A_prime

            y_t.append(Is)
            x_t.append(Js)

            steps+=1

    if steps<5000:
        print 'reached goal in %i steps, episode %i'% (steps, -1)
    else:
        print 'couldnt reach goal in 5000 steps with greedy'

    plt.imshow(wind, interpolation='none')
    plt.plot(np.asarray(x_t), np.asarray(y_t),'o-')
    plt.scatter(x_t[-1], y_t[-1], color='red', s=50)

    plt.savefig('sample_iter_%02i.png' % it)

################################################################################ 
# MAIN LOOP

START = 1
GOAL = 2

GAMMA = 1
EPSILON = 0.1
ALPHA = 0.5


EPISODES = 1500


KINGS = False
STAY = False # wether not moving is an option
STOCHASTIC = False # stochastic wind

wind = np.genfromtxt('wind.txt', delimiter=1).astype(int)
grid = np.genfromtxt('grid.txt', delimiter=1)


if KINGS:
    actions = [0, 1, 2, 3, 5, 6, 7, 8]
else:
    actions = [1, 3, 5, 7]

if STAY:
    actions.append(4)

actions = np.asarray(actions).astype(int)

H, W = wind.shape

Q = np.zeros((H,W,9))
        
Q = sarsa(Q, EPISODES, ALPHA, GAMMA, EPSILON, STOCHASTIC)


plot_sample(Q,1, STOCHASTIC)
