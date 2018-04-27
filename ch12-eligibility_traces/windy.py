""" Windy Gridworld Problem Using Eligibility Traces (Sutton and Barto 2nd edition, Chapter 12)

Includes REINFORCE (with and without baseline) and Actor-Critic
State representation can be a 3rd order polynomial on position or a one-hot vector.



April 18, 2018

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

def true_online_sarsa_lambda(w, alpha=0.125, lambda_=0.9, gamma=1.0):
    """ Sutton and Barto 2nd Edition, Section 12.7, Page 252 """

    global grid, GOAL, EPISODES, MAX_STEPS


    counts = []
    for ep in range(EPISODES):
        Is, Js = np.where(grid==START)

        Is = Is[0]
        Js = Js[0]


        epsilon = 0.1#max( 0.2*(1-ep/float(EPISODES))**2, 0.1)

        
        a = epsilon_greedy(Is, Js, epsilon=epsilon)

        
        xx = x(Is, Js, a)
        
        z = np.zeros_like(xx)

        Q_old = 0
        
        count = 0 

        ep_s = []
        ep_R = []

        

        while grid[Is, Js] != GOAL and (count < MAX_STEPS):
            count+=1

            R, Is_prime, Js_prime = step(Is, Js, a)
            a_prime = epsilon_greedy(Is_prime, Js_prime, epsilon=epsilon)

            
            xx_prime = x(Is_prime, Js_prime, a_prime)

            Q = np.dot(w,xx)
            Q_prime = np.dot(w,xx_prime)


            delta = R + gamma * Q_prime - Q


            
            z = gamma * lambda_ * z + (1 - alpha * gamma * lambda_ * np.dot(z,xx)) * xx
                         
            w_update = alpha * (delta + Q - Q_old) * z - alpha * (Q-Q_old) * xx

            # w_update = alpha* delta * xx

            #print np.max(np.abs(w_update))#, delta, 'Q', Q, [Is, Js],a , 'Q_prime',  Q_prime, [Is_prime, Js_prime], a_prime

            w += w_update

            Q_old = Q_prime
            
            Is = Is_prime
            Js = Js_prime
            a = a_prime

            xx = x(Is, Js, a)

            ep_s.append([Is, Js])
            

        counts.append(count)
        if ep % 100 == 0:
            print 'ep %i, count: %i' % (ep, count) #, ep_s

    # finished, plot steps per episode
    plot_curve(counts, 'true_online_sarsa_lambda_returns--alpha_%2.2e--lambda_%2.2e' % (alpha, lambda_))
################################################################################
## AUX FUNCTIONS
################################################################################
def epsilon_greedy(I,J, epsilon=0.1):
    if np.random.rand() <  epsilon:
        # return random action
        return np.random.choice(9)

    max_q = -np.inf
    max_a = -1


    for a in range(9):
        if q(w,I,J,a) > max_q:
            max_q = q(w,I,J,a)
            max_a = a
    return max_a
        

def step(Is, Js, a, stochastic_wind=0):
    """ do one step in windy gridworld """
    global grid, GOAL

    vx, vy = action_to_pair(a)

    Is_prime = Is + vy - wind[Is,Js] + stochastic_wind * ( wind[Is,Js]>0) * np.random.choice([-1,0,1])
    Js_prime = Js + vx

    Is_prime = min(H-1, max(0,Is_prime))
    Js_prime = min(W-1, max(0,Js_prime))
    

    if grid[Is_prime, Js_prime] == GOAL:
        R = 0
    else:
        R = -1

    return R, Is_prime, Js_prime 

################################################################################
def x(I,J,a):
    """ one-hot vector for SxA... super inefficient """

    global grid, dA, GOAL
    

    H, W = grid.shape

    xx = np.zeros(H*W*dA) 

    if grid[I,J] == GOAL:
        return xx
    else:
        xx[I*W + J + H*W*a] = 1
        return xx




def v(w,I,J):
    """ See Sutton and Barto 2nd edition 13.4, page 273 """
    global grid, GOAL

    if grid[I,J] == GOAL:
        return 0

    return np.dot(w,x(I,J))
    
def q(w,I,J,a):
    return np.dot(w,x(I,J,a))


# non-king moves: 1, 3, 5, 7
# king moves: 0, 1, 2, 3, 5, 6, 7, 8
################################################################################
def action_to_pair(a):
    assert(a>=0 and a<9)
    vx = int(np.floor(a/3)-1)
    vy = int(np.mod(a,3)-1)
    return vx, vy

def pair_to_action(vx, vy):
    assert np.abs(vx)<=1 and np.abs(vy)<=1

    return int((vx+1)*3 + vy + 1)


# ################################################################################
# def plot_sample(Q, it, stochastic_wind, eps=0):
#     global wind

#     plt.clf()
#     x_t = []
#     y_t = []

#     H, W = wind.shape
    
#     Is, Js = np.where(grid==START)
            
#     A, vx, vy = epsilon_greedy(Q, Is, Js, eps)

#     steps = 0

#     y_t.append(Is)
#     x_t.append(Js)
        
#     while grid[Is, Js] != GOAL and steps <5000:
#             # take action A
#             Is_prime = Is + vy - wind[Is,Js] + stochastic_wind * ( wind[Is,Js]>0) * np.random.choice([-1,0,1])
#             Js_prime = Js + vx
    
#             Is_prime = min(H-1, max(0,Is_prime))
#             Js_prime = min(W-1, max(0,Js_prime))
    
#             # choose A_prime from S_prime
#             A_prime, vx_prime, vy_prime = epsilon_greedy(Q, Is_prime, Js_prime, eps)
    
    
#             Is = Is_prime
#             Js = Js_prime

#             vx = vx_prime
#             vy = vy_prime
#             A = A_prime

#             y_t.append(Is)
#             x_t.append(Js)

#             steps+=1

#     if steps<5000:
#         print 'reached goal in %i steps, episode %i'% (steps, -1)
#     else:
#         print 'couldnt reach goal in 5000 steps with greedy'

#     plt.imshow(wind, interpolation='none')
#     plt.plot(np.asarray(x_t), np.asarray(y_t),'o-')
#     plt.scatter(x_t[-1], y_t[-1], color='red', s=50)

#     plt.savefig('sample_iter_%02i.png' % it)


def print_value_img(w,counts, ep):
    global grid
    H, W = grid.shape

    value = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            value[i,j] = v(w, i, j)
    
    plt.clf()
    plt.imshow(value, interpolation='none')
    plt.colorbar()
    plt.savefig('value_imgs/value_%06i.png' % ep)

    plt.clf()
    plt.imshow(counts, interpolation='none')
    plt.colorbar()
    plt.savefig('value_imgs/counts_%06i.png' % ep)


def plot_curve(a, title):
    a = np.asarray(a)
    plt.clf()
    plt.plot(a)
    plt.title(title)
    os.system('mkdir -p figures')
    plt.savefig('figures/%s.png' % title.replace(' ', '_'))




################################################################################ 
# MAIN LOOP

START = 1
GOAL = 2



EPISODES = 10000
MAX_STEPS = 100


KINGS = True
STAY = False # wether not moving is an option
STOCHASTIC = False # stochastic wind

wind = np.genfromtxt('wind.txt', delimiter=1).astype(int)
grid = np.genfromtxt('grid.txt', delimiter=1)


if KINGS:
    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
else:
    actions = [1, 3, 5, 7]

if STAY:
    actions.append(4)

actions = np.asarray(actions).astype(int)

H, W = wind.shape


##
## Initialize parameters

# type of representation: 3rd degree polynomial or one-hot vector (indicator)
#REPR = 'polynomial'
REPR = 'indicator'


dA = 9 # dimension of possible actions

w = np.zeros_like(x(0,0,0))

wind *= 0


#print grid

if __name__ == '__main__':

    true_online_sarsa_lambda(w)


