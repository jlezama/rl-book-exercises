""" Jack's Car Rental Problem. Exercise 4.5, Sutton and Barto 2nd edition.
jlezama@fing.edu.uy
"""
import os
import numpy as np
from scipy.stats.distributions import poisson
import matplotlib
import matplotlib.pyplot as plt

def policy_evaluation(V, pi, PR, gamma):
    """
    V should be a dict containing values for all states
    pi should be a dict containing the chosen action for each state
    P contains the transition probabilities P(s'|s,a)
    R contains the reward function R(s',s,a)
    gamma is the discount factor
    """

    global MAX_CARS, THETA



    Delta = np.inf

    while Delta > THETA:
        print 'entering while'
        Delta = 0
        
        for s1 in range(MAX_CARS):
            for s2 in range(MAX_CARS):
        
                print 'value evaluation for state %i,%i' % (s1, s2)
                v = V[s1,s2]
                a = pi[s1,s2]
                V[s1,s2] = 0
                for s1_prime in range(MAX_CARS):
                    for s2_prime in range(MAX_CARS):
                        # V[s] += P[s,a,s_prime]*(R[s,a,s_prime] + gamma*V[s_prime]) # dictionary version
                        P_sas_prime, R_sas_prime = PR((s1,s2), a, (s1_prime, s2_prime)) # funciton version
                        V[s1, s2] += P_sas_prime*(R_sas_prime + gamma*V[s1_prime, s2_prime]) # function version
                Delta = max(Delta, abs(v - V[s1, s2]))
                print Delta
        
    return V

def policy_improvement(V, pi, actions, PR, gamma):
    policy_stable = True


    for s1 in range(MAX_CARS):
        for s2 in range(MAX_CARS):
    
            print 'policy improvement for state %i/%i' % (s1, s2)
            old_action = pi[s1, s2]
    
            max_return = -np.inf
            argmax_a = -np.inf
    
            for a in actions:
                expected_return = 0
                for s1_prime in range(MAX_CARS):
                    for s2_prime in range(MAX_CARS):

                    
                        P_sas_prime, R_sas_prime = PR((s1,s2), a, (s1_prime, s2_prime)) # funciton version
                        expected_return += P_sas_prime*(R_sas_prime + gamma*V[s1_prime, s2_prime]) # function version
                    
                if expected_return > max_return:
                    max_return = expected_return
                    argmax_a = a
                
            pi[s1, s2] = argmax_a
            
            if old_action != pi[s1, s2]:
                policy_stable = False
        
    return pi, policy_stable
                


################################################################################
# REWARD AND TRANSITION PROBABILITIES
################################################################################
def PR(s, a, s_prime):
    global lambda_ret1, lambda_req1, lambda_ret2, lambda_req2, MAX_TRIPS

    if np.abs(a)>MAX_TRIPS:
        # maximum 5 cars returned
        return 0
    morning_loc1 = s[0] + a
    morning_loc2 = s[1] - a

    night_loc1 = s_prime[0]
    night_loc2 = s_prime[1]

    P1, R1 = prob_ret_req(morning_loc1, night_loc1, lambda_ret1, lambda_req1)
    P2, R2 = prob_ret_req(morning_loc2, night_loc2, lambda_ret2, lambda_req2)
    
    P = P1 * P2
    R = R1 + R2 - np.abs(a)*2

    return P, R

def prob_ret_req(n_morning, n_night, lambda_ret, lambda_req):
    """ 
    Probability for one agency of having n_morning cars in the morning and
    n_night cars in the night. Depends on the probabilities of returns and
    requests, as well as the max car availability.
    """
    prob = 0
    difference = n_night - n_morning
    R = 0

    for ret in range(int(10*lambda_ret)):
        for req in range(int(10*lambda_req)):
            if ret-req != difference:
                continue
            p_ret = poisson.pmf(ret, lambda_ret)
            p_req = poisson.pmf(req, lambda_req)
            


            prob += p_ret*p_req

            R += p_ret * p_req * req * 10  # expected reward

    return prob, R

def plot(V, pi, it):
    os.system("mkdir -p figures")
    
    fig, axes = plt.subplots(1, 2)
    
    ax = axes[0]
    im = ax.imshow(V, interpolation='none')
    ax.set_title('V')
    ax.set_xlabel('Location 1')
    ax.set_ylabel('Location 2')
    
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(pi, interpolation='none')
    ax.set_title('pi')
    ax.set_xlabel('Location 1')
    ax.set_ylabel('Location 2')
    
    plt.colorbar(im, ax=ax)
    plt.savefig('figures/result_iter_%02i.png' % it)

################################################################################
# MAIN LOOP 
################################################################################

acc_factor = 2.0

THETA = 5.0
MAX_CARS = int(20/acc_factor)
MAX_TRIPS = int(5/acc_factor)

# DEFINE PARAMETERS

actions = range(-1*MAX_TRIPS, MAX_TRIPS+1)

V = np.zeros((MAX_CARS, MAX_CARS))
pi = np.zeros((MAX_CARS, MAX_CARS))


gamma = 0.9

lambda_ret1 = 3/acc_factor
lambda_ret2 = 2/acc_factor
lambda_req1 = 3/acc_factor
lambda_req2 = 4/acc_factor


# RUN ITERATIONS
policy_stable = False

it = 0

plot(V, pi, it)

while not policy_stable:
    V = policy_evaluation(V, pi, PR, gamma)
    pi, policy_stable = policy_improvement(V, pi, actions, PR, gamma)
    it += 1
    plot(V, pi, it)
