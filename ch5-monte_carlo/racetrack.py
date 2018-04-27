""" Racetrack Problem. Exercise 5.8, Sutton and Barto 2nd edition.
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
def on_policy_mc_control(Q,C,pi):
    """ R contains returns """
    global gamma, EPISODES, EPSILON

    for episode in range(EPISODES):

        if (episode % 10000)==0:
            print 'episode %i of %i' % (episode, EPISODES)
            plot_sample(pi, episode)
            plot(Q, pi, episode)
            EPSILON = max(0.1, EPSILON*.95)
            print 'new EPSILON', EPSILON



        S, A, R = generate_episode(pi)
        T = len(S)

        appeared = dict()
                
        for t in range(T-1):
            Rt = R[t]
            St = S[t]
            At = A[t]

            if not (St,At) in appeared.keys():
                appeared[(St,At)] = True
                #print 'first (s, a) occurrence'
            else:
                #print '(s, a) already occurred'
                continue
                            

            
            x_t = St[0]
            y_t = St[1]
            vx_t = St[2]
            vy_t = St[3]
    
            # compute return (should be only for first appearance of S,A, TODO)
            Gt = 0
            for i in range(t,T-1):
                Gt += (gamma**(i-t))*R[i]
            

            # print y_t, x_t, vx_t, vy_t,At
            C[y_t, x_t, vx_t, vy_t,At] += 1
            Q[y_t, x_t, vx_t, vy_t,At] += 1/C[y_t, x_t, vx_t, vy_t,At]*(Gt - Q[y_t, x_t, vx_t, vy_t,At])



        for t in range(T):
            St = S[t]
            x_t = St[0]
            y_t = St[1]
            vx_t = St[2]
            vy_t = St[3]

            best_actions = np.where(Q[y_t,x_t,vx_t,vy_t,:]==np.max(Q[y_t,x_t,vx_t,vy_t,:]))[0]
            if len(best_actions)>1:
                best_action = np.random.choice(best_actions)
            else:
                best_action = best_actions[0]
            
            pi[y_t, x_t, vx_t, vy_t] = best_action



    return Q, pi

def off_policy_mc_control(Q, C, pi):
    global gamma, EPISODES


    for iter in range(EPISODES):
        S, A, R = generate_episode()
        G = 0
        W = 1
    
        T = len(S)
    
        for t in reversed(range(T)):
            Rt = R[t-1]
            St = S[t-1]
            At = A[t-1]

            #print 'Rt', Rt, t, T

    
            x_t = St[0]
            y_t = St[1]
            vx_t = St[2]
            vy_t = St[3]
    
            G = gamma*G + Rt
            C[y_t,x_t,vx_t,vy_t,At] += W

            update = Q[y_t,x_t,vx_t,vy_t,At] + W/C[y_t,x_t,vx_t,vy_t,At] * ( G - Q[y_t,x_t,vx_t,vy_t,At] )
            #print 'update', update

            #print 'before', Q[y_t,x_t,vx_t,vy_t,:]
            Q[y_t,x_t,vx_t,vy_t,At] = update

            #print 'after', Q[y_t,x_t,vx_t,vy_t,:], At

            max_action = np.max(Q[y_t,x_t,vx_t,vy_t,:])

            best_actions = np.where(Q[y_t,x_t,vx_t,vy_t,:]==max_action)[0]
            if len(best_actions)>1:
                best_action = np.random.choice(best_actions)
            else:
                best_action = best_actions[0]

            #print 'max_action', max_action, Q[y_t,x_t,vx_t,vy_t,:], 'At', At, 'best_actions', best_actions, 'best_action', best_action

            # if np.abs(Q[y_t,x_t,vx_t,vy_t,At] - Q[y_t,x_t,vx_t,vy_t,best_action])<1e0:
            #     best_action = At
            #print 'best_action', best_action, Q[y_t,x_t,vx_t,vy_t,:], 'At', At

            pi[y_t,x_t,vx_t,vy_t] = best_action

            if best_action != At:
                print 'breaking!', T-t
                break

            dx, dy = action_to_pair(At)

            if dx ==0: 
                pdx = 3/6.
            elif dx ==1:
                pdx = 2/6.
            else:
                pdx = 1/6.
            if dy == 1:
                pdy = 3/5.
            elif dy == 0:
                pdy = 1/5.
            elif dy ==-1:
                pdy = 1/5.
            

            W *= 1/(pdx*pdy) # TODO b is not random
        
    return Q, pi

def generate_episode(pi=None, eps=None, noise=True):
    global track, actions, MAX_T, EPSILON

    H, W = track.shape
    
    if eps is None:
        eps = EPSILON



    # first state: random start location, 0 velocity
    x_0, y_0 = random_start()



    S = [(x_0, y_0, 0, 0)]
    A = []
    R = []

    for t in range(MAX_T):
        St = S[-1]
        x_t = St[0]
        y_t = St[1]
        vx_t = St[2]
        vy_t = St[3]



        # Noise with probability 0.1 at each time step the velocity increments are both zero
        if noise and np.random.rand()<0.1:
            delta_vx = 0
            delta_vy = 0
            At1 = pair_to_action(delta_vx, delta_vy)
            
            
        elif pi is None:
            # \epsilon-soft policy b
            delta_vx = np.random.choice([-1, 1, 1, 0, 0, 0]) # choose mostly no horiz accel
            delta_vy = np.random.choice([-1, 0, 1, 1, 1]) # choose mostly vert accel 
            At1 = pair_to_action(delta_vx, delta_vy)

        else:
            At1 = pi[y_t,x_t,vx_t,vy_t] if np.random.rand()>eps else np.random.randint(9)
            delta_vx, delta_vy = action_to_pair(At1)

        
        vx_t1 = max(0,min(MAX_SPEED, vx_t + delta_vx))
        vy_t1 = max(0,min(MAX_SPEED, vy_t + delta_vy))

        if vx_t1==0 and vy_t1==0:
            #print 'both zero!',t, delta_vx, delta_vy, At1
            if np.random.rand()>0.5:
                vx_t1 = 1
            else:
                vy_t1 = 1
            At1 = pair_to_action(vx_t1, vy_t1) # should be 0,1 or 1,0
            assert (vx_t1+vy_t1)==1
                
        x_t1 = x_t + vx_t1
        y_t1 = y_t - vy_t1 # vertical is negative to go up in matrix


        # check if it went over boundary
        touched_boundary = False


        if x_t1 >= W or x_t1 < 0 or y_t1 >= H or y_t1 <0 or track[y_t1, x_t1] == BOUNDARY:
            touched_boundary = True


        for vxx in range(vx_t1):
            if touched_boundary:
                 break
            for vyy in range(vy_t1):
                if track[y_t - vyy,x_t+vxx] == BOUNDARY:
                    touched_boundary = True
                    break


        if touched_boundary:
            x_t1, y_t1 = random_start()
            vx_t1 = 0
            vy_t1 = 0



        St1 = (x_t1, y_t1, vx_t1, vy_t1)
        
            
        Rt1 = -1


        S.append(St1)
        A.append(At1)
        R.append(Rt1)

        terminate = False
        if track[y_t1, x_t1] == FINISH:
            print 'FINISHED in %i steps!' % t
            terminate = True
            break
        # print St1, At1, Rt1
    if not terminate:
        print 'didnt make it to the end ----------', eps
    return S, A, R


################################################################################
# AUX FUNCTIONS

def action_to_pair(a):
    assert(a>=0 and a<9)

    vx = int(np.floor(a/3)-1)
    vy = int(np.mod(a,3)-1)


    return vx, vy

def pair_to_action(vx, vy):
    assert np.abs(vx)<=1 and np.abs(vy)<=1

    return int((vx+1)*3 + vy + 1)

def random_start():
    
    global track

    # possible start positions
    Is, Js = np.where(track==START)
    ix_start = np.random.randint(len(Is))
    return  Js[ix_start], Is[ix_start] # horizontal coord first


def plot(Q, pi, it):
    os.system("mkdir -p figures")
    
    fig, axes = plt.subplots(MAX_SPEED+1, MAX_SPEED+2)
    
    ax = axes[0,0]

    im = ax.imshow(np.mean(Q, axis=(2,3,4)), interpolation='none')
    ax.set_title('Q')

    
    plt.colorbar(im, ax=ax)

    count = MAX_SPEED+3
    for vx in range(MAX_SPEED+1):
        for vy in range(MAX_SPEED+1):

            ax = axes[vx, 1+vy]
            im = ax.imshow(pi[:,:,vx,vy], interpolation='none')
            ax.set_title('pi (vx: %i, vy: %i)' % (vx, vy))
            count += 1
    
    plt.colorbar(im, ax=ax)
    plt.savefig('figures/result_iter_%09i.png' % it)

def plot_old(Q, pi, it):
    os.system("mkdir -p figures")
    
    fig, axes = plt.subplots(1, 2)
    
    ax = axes[0]
    im = ax.imshow(np.mean(Q, axis=(2,3,4)), interpolation='none')
    ax.set_title('Q')
    ax.set_xlabel('Location 1')
    ax.set_ylabel('Location 2')
    
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(np.mean(pi, axis=(2,3)), interpolation='none')
    ax.set_title('pi')
    ax.set_xlabel('Location 1')
    ax.set_ylabel('Location 2')
    
    plt.colorbar(im, ax=ax)
    plt.savefig('figures/result_iter_%09i.png' % it)

def plot_sample(pi, it):
    os.system('mkdir -p samples')
    global track

    H, W = track.shape

    S, A, R = generate_episode(pi, eps=0, noise=False)

    f = open('samples/sample_iter_%09i.txt' % it, 'w')
    f.write(str(S))
    f.close()

    plt.clf()
    x_t = []
    y_t = []

    for St in S:
        x_t.append(St[0])
        y_t.append(St[1])

    plt.imshow(track, interpolation='none')
    plt.plot(np.asarray(x_t), np.asarray(y_t),'o-')
    plt.scatter(x_t[-1], y_t[-1], color='red', s=50)

    plt.savefig('samples/sample_iter_%09i.png' % it)


################################################################################ 
# MAIN LOOP


START = 2
FINISH = 3
TRACK = 1
BOUNDARY = 0

gamma = 0.9
accelerate_factor = 2.0

MAX_SPEED = 4
MAX_T = 500 # Max episode length
EPISODES = int(1e5)
EPSILON = 0.5

track = np.genfromtxt('racetrack.txt', delimiter=1)

actions = range(9) # cartesian product of (-1, 0, 1)

H, W = track.shape

Q = np.ones((H,W,MAX_SPEED+1,MAX_SPEED+1,9))*(-1*MAX_T) # pessimism in the face of uncertainty :-)
C = np.zeros((H,W,MAX_SPEED+1,MAX_SPEED+1,9))


pi = np.random.randint(low=0, high=9, size=(H,W,MAX_SPEED+1,MAX_SPEED+1))

Q, pi = on_policy_mc_control(Q,C,pi)


plot(Q, pi, 0)

# plot a few more samples episodes
for i in range(10):
    plot_sample(pi, EPISODES+i)

