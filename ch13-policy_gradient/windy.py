""" Windy Gridworld Problem Using Policy Gradient (Sutton and Barto 2nd edition, Chapter 13)

Includes REINFORCE (with and without baseline) and Actor-Critic
State representation can be a 3rd order polynomial on position or a one-hot vector.



April 11, 2018

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
def actor_critic(w, theta, alpha_w=1e-3, alpha_theta=1e-2, gamma=0.99):
    global GOAL, EPISODES, MAX_STEPS

    Gs = []

    count = 0
    for episode in range(EPISODES):


        
        if episode % 100 ==0:
            print 'NEW EPISODE %i/%i! (%i)' % (episode, EPISODES, count),  compute_pi(theta, 3, 6), action_to_pair(np.random.choice(9, p=compute_pi(theta, 3, 6)))

            print 'values', v(w,3,0), v(w,3,1), v(w,3,2), v(w,3,3), v(w,3,4), v(w,3,5), v(w,3,6), v(w,3,7), v(w,3,8), v(w,3,9)


        count = 0

        counts = np.zeros_like(grid)
        
        Is, Js = np.where(grid==START)
        II = 1


        G = 0


        while grid[Is, Js] != GOAL and count < MAX_STEPS:

            pi = compute_pi(theta, Is, Js)
            
            #print 'pi', pi, [Is, Js]

            a = np.random.choice(9, p=pi)

            #a = 7

            # if 1:#np.random.rand()<1.0:
            #     a = np.random.choice(9)
            
            R, Is_prime, Js_prime = step(Is, Js, a)

            G += R
            
            vx, vy = action_to_pair(a)
            
            delta = R + gamma * v(w, Is_prime, Js_prime) -v(w, Is, Js)

            w_update = alpha_w * II * delta * x(Is,Js)
            
            w += w_update

            theta_update =  alpha_theta * II * delta * compute_grad(theta, Is, Js, a)
            theta += theta_update




            # print 'moving from [%i %i] to [%i %i] (%i: %i, %i)' % (Is, Js, Is_prime, Js_prime,a, vx, vy)
            # print 'pi', pi
            # print 'grad', compute_grad(theta, Is, Js, a)[3:]
            # # print 'delta', delta
            #print 'theta_update', theta_update
            # print '-----'

            #raise

            counts[Is, Js] += 1
            
            II *= gamma
            Is = Is_prime
            Js = Js_prime

            count +=1

            if count % 1000==0:
                print 'still computing', count
                print 'moving from [%i %i] to [%i %i] (%i: %i, %i)' % (Is, Js, Is_prime, Js_prime,a, vx, vy)
                print 'pi', pi
                print 'grad', compute_grad(theta, Is, Js, a)
                # print 'delta', delta
                print 'theta_update', theta_update
                print 'w_update', w_update
                print '-----'
        
        Gs.append(G)
        
        if episode % 1000 == 0:
            print_value_img(w, counts, episode)

    plot_curve(Gs, 'actor_critic_returns--alpha_theta_%2.2e--alpha_w_%2.2e--_%s' % (alpha_theta, alpha_w, REPR))


################################################################################
################################################################################
def REINFORCE(theta, w, gamma=1.0, alpha_theta=1e-5, alpha_w=1e-7):
    global grid, GOAL, EPISODES, MAX_STEPS, REPR

    H,W = grid.shape

    counts = np.zeros((H,W))

    G_0s = []
    for ep in range(EPISODES):


        Is, Js = np.where(grid==START)


        Is = Is[0]
        Js =Js[0]


        pi = compute_pi(theta,Is, Js)
        a = np.random.choice(9, p=pi)


        counts *= 0
        
#        a = 7


        ep_s = []
        ep_a = []
        ep_R = []

        ep_s.append([Is, Js])
        ep_a.append(a)

        R, Is, Js = step(Is, Js, a)
        

        #print Is, Js, a, action_to_pair(a)
        

        ep_R.append(R)
        

        while grid[Is, Js] != GOAL and len(ep_s)<MAX_STEPS:

            pi = compute_pi(theta, Is, Js)


            a = np.random.choice(9, p=pi)



            ep_s.append([Is, Js])
            ep_a.append(a)


            if len(ep_s) % 1000 == 0:
                print len(ep_s), Is, Js, pi,a, action_to_pair(a)

            # if np.random.rand()<0.05:
            #      a = np.random.choice(9)
            
           # a= 7

            R, Is, Js = step(Is, Js, a)

            #print Is, Js

            counts[Is, Js]+=1
        
            ep_R.append(R)

        # print 'GOAL!'
        ep_R = np.asarray(ep_R)




        for t in range(len(ep_s)):
           G_t = np.sum(ep_R[(t):])
           Is_t, Js_t = ep_s[t][0], ep_s[t][1]
           
           delta = G_t - v(w, Is_t, Js_t)


           w_update = alpha_w * (gamma**t) * delta * x(Is_t, Js_t)
           

           w += w_update

           #print w_update

           # WITHOUT BASELINE:
           #theta_update =  alpha_theta * (gamma**t) * G_t * compute_grad(theta, Is_t, Js_t,ep_a[t])

           # WITH BASELINE:
           theta_update = alpha_theta * (gamma**t) * delta * compute_grad(theta, Is_t, Js_t,ep_a[t])

           theta += theta_update
           

        G_0 = np.sum(ep_R)

        if ep % 100 == 0:
            print 'ep %i, G_0 %f' % (ep, G_0),  compute_pi(theta, 3, 6), v(w, 3,0), v(w,3,1), v(w,3,2), v(w,3,3), v(w,3,4), v(w,3,5), v(w,3,6), v(w,3,7), v(w,3,8), v(w,3,9), w.shape
            print ep_s

        G_0s.append(G_0)
        
        if ep % 1000 == 0:
            print_value_img(w, counts, ep)


    plot_curve(G_0s, 'REINFORCE_returns--alpha_theta_%2.2e--alpha_w_%2.2e--_%s' % (alpha_theta, alpha_w, REPR))

    return theta, np.asarray(G_0s)



################################################################################
## AUX FUNCTIONS
################################################################################
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
def x(I,J):
    global REPR
    if REPR == 'polynomial':
        return x_polynomial(I,J)
    elif REPR == 'indicator':
        return x_indicator(I,J)
    else:
        raise ValueError('unknown representation type')

def x_indicator(I,J):
    global grid
    H, W = grid.shape

    xx = np.zeros(H*W) 
    xx[I*W+J] = 1
    
    return xx

def x_polynomial(I, J):
    # returns a vector representation of x
    global grid
    H,W = grid.shape
    

    xx = np.zeros(10)


    xx[0] = (I-H/2.)/float(H/2.)
    xx[1] = (J-W/2.)/float(W/2.)
    xx[2] = (xx[0])**2
    xx[3] = (xx[1])**2
    xx[4] = xx[0]*xx[1]

    xx[5] = xx[0]**3
    xx[6] = xx[1]**3
    xx[7] = xx[2]*xx[1]
    xx[8] = xx[3]*xx[0]

    
    xx[9] = 1 # bias term
    

    return xx


################################################################################
def compute_pi(theta, I, J):
    # compute soft-max for linear feature theta^T.x

    xx = x(I,J)

    scores = np.dot(xx.T, theta)

    scores -= np.max(scores)

    pi = np.exp(scores)/np.sum(np.exp(scores))

    assert np.abs(np.sum(pi)-1)<1e-9, np.sum(pi)

    return pi

################################################################################
def compute_grad(theta,I,J,a):
    # compute soft-max for linear feature theta^T.x

    global actions
    pi = compute_pi(theta,I, J)


    grad = np.zeros_like(theta)

    
    for b in actions:
        if b==a:
            grad[:,b] = x(I,J)*(1-pi[b])
        else:
            grad[:,b] = -1*pi[b]*x(I,J)
            
    #print '----'
    #print pi, grad[:,a],grad[:,a+1], x(I,J), a
    # raise

    return grad


def v(w,I,J):
    """ See Sutton and Barto 2nd edition 13.4, page 273 """
    global grid, GOAL

    if grid[I,J] == GOAL:
        return 0

    return np.dot(w,x(I,J))
    


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
dX = int(x(0,0).shape[0]) #H*W # dimension of x

theta = np.zeros((dX,dA))
w = np.zeros(dX)

wind *= 0


#print grid

if __name__ == '__main__':

    actor_critic(w, theta)

    #REINFORCE(theta,w)

