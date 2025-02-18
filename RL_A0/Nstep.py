#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        for t in range(done):
            m = min(n, done - t)
            new_Q_sa = 0
            if np.all(self.env._state_to_location(states[t + m]) == (7,3)):
                for i in range(m):
                    new_Q_sa += (self.gamma ** i) * rewards[t + i]
            else:
                for i in range(m):
                    new_Q_sa += (self.gamma ** i) * rewards[t + i]
                new_Q_sa += (self.gamma ** m) * max(self.Q_sa[states[t + m], ])
            self.Q_sa[states[t], actions[t]] = self.Q_sa[states[t], actions[t]] + self.learning_rate * (new_Q_sa - self.Q_sa[states[t], actions[t]])

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your n-step Q-learning algorithm here!
    while n_timesteps > 0:
        s = env.reset()
        states = [s]
        actions = []
        rewards = []
        # (1) Simulate a single episode from start state to the state determined by 'max_episode_length'
        for t in range(max_episode_length):
            n_timesteps = n_timesteps - 1
            a = pi.select_action(states[t], policy, epsilon, temp) # epsilon-greedy
            next_s, r , done = env.step(a)
            actions.append(a)
            states.append(next_s)
            rewards.append(r)
            # all_rewards.append(r)
            # if plot and n_timesteps < 100:
            #     env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.01)
            if done or n_timesteps == 0:
                break

        T_ep = t + 1
         # (2) Compute n-step targets and update
        pi.update(states=states, actions=actions, rewards=rewards, done=T_ep, n=n)
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
        
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()
