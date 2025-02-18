#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax
import random

class BaseAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'greedy':
             # choose the best action
            a = argmax(self.Q_sa[s])
            # TO DO: Add own code
            # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
            
        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            # if 1-epsilon*((self.n_actions-1)/self.n_actions) > epsilon/self.n_actions:
            # Initialize probabilities for each action with epsilon / |A|
            probabilities = np.ones(self.n_actions) * epsilon / self.n_actions
            # if random.random() <= epsilon:
            #     # choose the best action
            #     a = argmax(self.Q_sa[s])
            # else:
            #     # choose randomly
            #     a = np.random.randint(0,self.n_actions)
            # Find the greedy action (action with maximum Q-value)
            if isinstance(self.Q_sa, dict):
                q_values = [self.Q_sa.get((s, a), 0) for a in self.n_actions]
            else:
                q_values = self.Q_sa[s]
            
            greedy_action = np.argmax(q_values)
            
            # Add (1 - epsilon) to the probability of the greedy action
            probabilities[greedy_action] += (1.0 - epsilon)
            
            # Select action based on the probability distribution
            a = np.random.choice(self.n_actions, p=probabilities)
            # TO DO: Add own code
            # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
                 
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            # Get all possible actions and their Q-values for the current state
            # actions = [a for s, a in Q_values.keys() if s == state]
            q_vals = np.array([self.Q_sa[(s, a)] for a in self.n_actions])
            
            # Compute action probabilities using the Boltzmann distribution
            # Subtract max Q-value for numerical stability
            max_q = np.max(q_vals)
            exp_q = np.exp((q_vals - max_q) / temp)
            probs = exp_q / np.sum(exp_q)
            
            # Create dictionary mapping actions to their probabilities
            # action_probs = dict(zip(actions, probs))
            
            # Sample action according to computed probabilities
            a = np.random.choice(self.n_actions, p=probs)
            
            # return action, action_probs
                
            # TO DO: Add own code
            a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        print(a)  
        return a
        
    def update(self):
        raise NotImplementedError('For each agent you need to implement its specific back-up method') # Leave this and overwrite in subclasses in other files


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
