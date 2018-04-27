""" Dyna-Q Maze Exercise 8.4, based on Example 8.1 Sutton and Barto 2nd edition.
jlezama@fing.edu.uy

TODO: implement exploration bonus

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
def dynaq(Q, Model,  alpha, gamma,  eps, steps, planning_steps):
    global grid, START, GOAL, BLOCK

    H, W = grid.shape
    
    Is, Js = np.where(grid==START)

    goals = 0
    
    for step in range(steps):

        A = epsilon_greedy(Q, Is, Js, eps)
    
        vx, vy = action_to_pair(A)

        
        Is_prime = Is + vy 
        Js_prime = Js + vx
    
        Is_prime = min(H-1, max(0,Is_prime))
        Js_prime = min(W-1, max(0,Js_prime))

        if grid[Is_prime, Js_prime] == BLOCK:
            Is_prime = Is
            Js_prime = Js
        

        reached_goal = False
        if grid[Is_prime, Js_prime] == GOAL:
            R = 1
            reached_goal = True
            goals += 1
        else:
            R = 0

        # step (d)
        Q[Is, Js, A] = Q[Is, Js, A] + alpha * (R + gamma * np.max(Q[Is_prime, Js_prime,:]) -Q[Is, Js, A])

        # step (e)
        Model[Is, Js, A, 0] = R # reward
        Model[Is, Js, A, 1] = Is_prime # s'
        Model[Is, Js, A, 2] = Js_prime                                       


        
        # step (f)
        # get visited states
        Is_visited, Js_visited, A_visited = np.where(Model[:,:,:,1] >= 0) # -1 is default unvisited state
        for n in range(planning_steps):
            ix = np.random.randint(Is_visited.shape[0])
            Is_n = Is_visited[ix]
            Js_n = Js_visited[ix]
            A_n = A_visited[ix]

            R_n = Model[Is_n, Js_n, A_n, 0]
            Is_prime_n = Model[Is_n, Js_n, A_n, 1]
            Js_prime_n = Model[Is_n, Js_n, A_n, 2]

            Q[Is_n, Js_n] = Q[Is_n, Js_n] + alpha * (R_n + gamma * np.max(Q[Is_prime_n, Js_prime_n, :]) - Q[Is_n, Js_n, A_n])
    


        if reached_goal:
            Is, Js = np.where(grid==START)
        else:
            Is = Is_prime
            Js = Js_prime
            
        if step %1000 ==0:
            print 'step',  step, Is, Js, Is_prime, Js_prime, vy, vx, 'reached goal %i times' % goals
               

    
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

    return best_action

def action_to_pair(a):
    # very lazy way to implement this
    
    if a == 0:
        vy = -1
        vx = 0
    elif a == 1:
        vy = 1
        vx = 0
    elif a == 2:
        vy = 0
        vx = -1
    elif a == 3:
        vy = 0
        vx = 1
    else:
        raise ValueError('Invalid action')

    return vx, vy

def pair_to_action(vx, vy):
    if (-1, 0) == (vy, vx):
        return 0
    elif (1, 0) == (vy, vx):
        return 1
    elif (0, -1) == (vy, vx):
        return 2
    elif (0, 1) == (vy, vx):
        return 3
    else:
        raise ValuError('Invalid pair')
    


def plot_sample(Q, it,  eps=0):
    global grid

    plt.clf()
    x_t = []
    y_t = []

    H, W = grid.shape
    
    Is, Js = np.where(grid==START)
            
    A = epsilon_greedy(Q, Is, Js, eps)
    vx, vy = action_to_pair(A)
    
    steps = 0

    y_t.append(Is)
    x_t.append(Js)
        
    while grid[Is, Js] != GOAL and steps <5000:
            # take action A
            Is_prime = Is + vy 
            Js_prime = Js + vx
    
            Is_prime = min(H-1, max(0,Is_prime))
            Js_prime = min(W-1, max(0,Js_prime))

            if grid[Is_prime, Js_prime] == BLOCK:
                Is_prime = Is
                Js_prime = Js

            
            # choose A_prime from S_prime
            A_prime = epsilon_greedy(Q, Is_prime, Js_prime, eps)

            vx_prime, vy_prime = action_to_pair(A_prime)
    
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

    plt.imshow(grid, interpolation='none')
    plt.plot(np.asarray(x_t), np.asarray(y_t),'o-')
    plt.scatter(x_t[-1], y_t[-1], color='red', s=50)

    plt.savefig('sample_iter_%02i.png' % it)

################################################################################ 
# MAIN LOOP

START = 1
GOAL = 3
BLOCK = 2

GAMMA = 0.95
EPSILON = 0.1
ALPHA = 0.1


STEPS = 10000
PLANNING_STEPS = 10


STAY = False # wether not moving is an option

grid = np.genfromtxt('grid.txt', delimiter=1).astype(int)


actions = [0, 1, 2, 3] # up, down, left, right


actions = np.asarray(actions).astype(int)

H, W = grid.shape

Q = np.zeros((H,W,4))

Model = np.ones((H,W,4,3)).astype(int)*-1 # R (1) and S' (2) for every S, A

Q = dynaq(Q, Model, ALPHA, GAMMA, EPSILON, STEPS, PLANNING_STEPS)


plot_sample(Q,1)
