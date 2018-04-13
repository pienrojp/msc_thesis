# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:55:05 2018

@author: pienrojp
"""

import gym
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

def build_states(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))
    
def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]
    
    
class feature_transform:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4,0.4 , 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)
        
    def transform(self, observation):
        cart_pos, cart_vel, pole_angle, pole_vel =  observation
        return build_states([
            to_bin(cart_pos, self.cart_position_bins),
            to_bin(cart_vel, self.cart_velocity_bins),
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.pole_velocity_bins)
            ])

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer
        
        num_states = 10**env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
                
    def predict(self, s):
        x = self.feature_transformer.transform(s)
        return self.Q[x]
    
    def update(self, s,a,G):
        x = self.feature_transformer.transform(s)
        self.Q[x,a] += 0.001*(G- self.Q[x,a] )
        
    def sample_action(self, s , eps):
        if np.random.random()<eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)
            
def play(model, eps, gamma):
    observation = env.reset()
    done = False    
    totalreward = 0
    iters = 0
    while not done:
        action = model.sample_action(observation, eps)
        prev_observation = observation        
        observation, reward, done, info = env.step(action)
        
        totalreward += reward
        
        if done and iters<199:
            reward = -300
        
        G = reward + gamma*np.max(model.predict(observation))
        model.update(prev_observation, action, G)
        
        iters += 1 
    return totalreward
    
    
def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg=  np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running avg")
    plt.show()
    
env = gym.make('CartPole-v0')
ft = feature_transform()
model = Model(env, ft)
gamma=0.9

N=20000
totalrewards = np.empty(N)
for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    totalreward = play(model, eps, gamma)
    totalrewards[n]=totalreward
    if n%100==0:
        print("episode",n,"total reward:", totalreward, "eps:", eps)
print("avg reward for last 100:", totalrewards[-100:].mean())
print("total steps:", totalrewards.sum())

plt.plot(totalrewards)
plt.title("Rewards")
plt.show()


plot_running_avg(totalrewards)

    
            