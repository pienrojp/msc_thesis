# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:55:02 2018

@author: pienrojp
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
from gym import wrappers

def get_action(s,w):
    return 1 if s.dot(w) > 0 else 0
    
def play(env, params, render=False):
    observation = env.reset()
    done = False
    t=0
    
    while not done and t < 10000:
             
        if render:
            env.render()
            t += 1
            action = get_action(observation, params)
            observation, reware, done, info = env.step(action)
            if done:
                break
            
        else:
            t += 1
            action = get_action(observation, params)
            observation, reware, done, info = env.step(action)
            if done:
                break
        
    return t

    
def play_many_time(env, T, params,render=False):
    episode_len = np.empty(T)
    
    for i in range(T):
        episode_len[i] = play(env,params,render)
    
    avg_len = episode_len.mean()
    print("avg lenght:",avg_len)
    return avg_len
 
     
    
def random_search(env):
    episode_len =  []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4)*2-1
        avg_len = play_many_time(env, 100, new_params)        
        episode_len.append(avg_len)
        
        if avg_len > best:
            params = new_params
            best = avg_len
            
    return episode_len, params

#play        
env = gym.make('CartPole-v0')
episode_len, params = random_search(env)
#plot result
plt.plot(episode_len)
plt.show()

#play other time
print("*** Best performance *** ")
play_many_time(env, 100, params, render=True)





    