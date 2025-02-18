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
# from Helper import argmax

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        # using loss function update
        self.Q_sa[s][a] = self.Q_sa[s][a] + self.learning_rate * (r + self.gamma * np.argmax(self.Q_sa[s_next]) - self.Q_sa[s][a])
        # TO DO: Add own code
        # print(self.Q_sa[s][a])

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    for n in range(n_timesteps):
        # sample initial state
        t_r = 0
        s = eval_env.reset()
        a = agent.select_action(s, 'egreedy', epsilon=epsilon, temp=temp)
        s_next, r, done = eval_env.step(a)
        agent.update(s, a, r, s_next, done)
        s = s_next
        t_r += r
        # Q learning
        while not done:   # budget?
            a = agent.select_action(s, 'egreedy', epsilon=epsilon, temp=temp)
            s_next, r, done = eval_env.step(a)
            agent.update(s, a, r, s_next, done)
            t_r += r
            if done:
                # reset state
                s = eval_env.reset()
            else:
                s = s_next
        print(t_r)
        if n % eval_interval == 0:
            meean_reward = agent.evaluate(eval_env=eval_env)
            eval_timesteps.append(n)
            eval_returns.append(meean_reward)
        if plot:
            env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1)
    
    # TO DO: Write your Q-learning algorithm here!
    
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution


    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    
    n_timesteps = 1000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
