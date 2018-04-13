# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 12:08:01 2018

@author: pienrojp
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from td0_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def epsilon_greedy(action, eps=0.1):
    p = np.random.random()
    if p>eps:
        return action
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
        

def max_dict(d):
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

        
grid = negative_grid(step_cost=-0.1)

print("rewards:")
print_values(grid.rewards, grid)

#get states
states = grid.all_states()

#initialize Q    
Q = {}
for s in states:
    Q[s] = {}
    for a in ALL_POSSIBLE_ACTIONS:
        Q[s][a] = 0
        
#How many times Q[s] has been updated
update_counts = {}
update_counts_sa = {}
for s in states:
    update_counts_sa[s]={}
    for a in ALL_POSSIBLE_ACTIONS:
        update_counts_sa[s][a] = 1.0
    
#repeat until converge
t=1.0
deltas = []
for it in range(10000):
    if it%100 ==0:
        t += 10e-2
    if it%2000 ==0:
        print("iter:", it)
        
    s = (2,0)
    grid.set_state(s)
    
    a = max_dict(Q[s])[0]
    a = random_action(a, eps=0.5/t)
    biggest_change = 0
    
    while not grid.game_over():
        r = grid.move(a)
        s2 = grid.current_state()
        
        a2 = max_dict(Q[s2])[0]
        a2 = random_action(a2, eps=0.5/t)
        
        alpha = ALPHA / update_counts_sa[s][a]
        update_counts_sa[s][a] += 0.005

        old_qsa = Q[s][a]
        Q[s][a] = Q[s][a] + alpha*(r + GAMMA*Q[s2][a2]-Q[s][a])        
        biggest_change = max(biggest_change, np.abs(old_qsa-Q[s][a]))
        
        update_counts[s] = update_counts.get(s,0) + 1
        
        s=s2
        a=a2
        
    deltas.append(biggest_change)
    
plt.plot(deltas)
plt.show()

policy = {}
V = {}

for s in grid.actions.keys():
    a, max_q = max_dict(Q[s])
    policy[s]=a
    V[s] = max_q
    
      # what's the proportion of time we spend updating each part of Q?
print("update counts:")
total = np.sum(list(update_counts.values()))
for k, v in update_counts.items():
  update_counts[k] = float(v) / total
print_values(update_counts, grid)

print("values:")
print_values(V, grid)
print("policy:")
print_policy(policy, grid)
    
        
        
    
    
