#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        # a = argmax(self.Q_sa[s])
        # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        a = np.argwhere(self.Q_sa[s, ] == np.amax(self.Q_sa[s, ]))
        a = a.flatten()
        if a.shape[0] > 1:
            a = np.random.choice(a.flatten()) # If there are multiple maxima
            return a
        elif a.shape[0] == 1:
            return a[0]
        # return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        new_Q_sa = 0
        next_states = np.where(p_sas != 0)[0]    # select the possible next state
        for s_next in next_states:
            new_Q_sa += p_sas[s_next]*(r_sas[s_next] + self.gamma * np.argmax(self.Q_sa[s_next]))
        self.Q_sa[s, a] = new_Q_sa
        # print(new_Q_sa)
        # pass
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
 
     # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    DP_error = 0
    i = 0
    while True:
        for s in range(QIagent.n_states):
            for a in range(QIagent.n_actions):
                p_sas, r_sas = env.model(s, a)
                # store current estimate
                x = QIagent.Q_sa[s, a]
                # update the Q valvue
                QIagent.update(s, a, p_sas, r_sas)
                # update max error
                DP_error = max(DP_error, abs(x - QIagent.Q_sa[s, a]))
                # print(DP_error)
        if DP_error < threshold :
            break
        # print(s)
        i += 1
        print("Q-value iteration, iteration {}, max error {}".format(i, DP_error))
                
    # Plot current Q-value estimates & print max error
    # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
 
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # view optimal policy
    done = False
    s = env.reset()
    rewards = []
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        rewards.append(r)
        # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    mean_reward_per_timestep = np.mean(rewards) #
    # TO DO: Compute mean reward per timestep under the optimal policy
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    
if __name__ == '__main__':
    experiment()
