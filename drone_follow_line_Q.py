# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:24:22 2018

@author: pienrojp
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:39:32 2018

@author: pienrojp
"""
from __future__ import print_function, division
from builtins import range
import numpy as np

import sys
sys.path.insert(0, '/home/pienrojp/AirSim/PythonClient')
from AirSimClient import *

''' Simple State discretize --> x=5 * y=10 gridworld'''

def build_state(features):
  return int("".join(map(lambda feature: str(int(feature)), features)))
build_state([1,1])
def to_bin(value, bins):
  return np.digitize(x=[value], bins=bins)[0]

x_bin, y_bin = 5, 12

class FeatureTransformer:
    def __init__(self):
        self.drone_position_x_bins = np.linspace(-15, 15, x_bin-1)  #5 bins for x axis on xy plane to goal
        self.drone_position_y_bins = np.linspace(0, dist_to_goal, y_bin-1)  #10 bins for y axis on xy plane to goal
        
    def transform(self, observation):
        drone_position_x = observation.x_val
        drone_position_y = observation.y_val
        return build_state([to_bin(drone_position_x, self.drone_position_x_bins), to_bin(drone_position_y, self.drone_position_y_bins) ])

    def transform_axis(self, observation):
        drone_position_x = observation[0]
        drone_position_y = observation[1]
        return build_state([to_bin(drone_position_x, self.drone_position_x_bins), to_bin(drone_position_y, self.drone_position_y_bins) ])



''' Action '''
possible_action = [1,2,3]

#5 discrete actions
def interpret_action(action):
    scaling_factor = 10
    quad_offset = (0, 0, 0)
    if action == 1:
        quad_offset = (scaling_factor, 0, 0)
    elif action == 2:
        quad_offset = (-scaling_factor, 0, 0)
    elif action == 3:
        quad_offset = (0, scaling_factor, 0)
    
    return quad_offset
''' Control '''

class Agent:
  def __init__(self, feature_transformer):
    self.feature_transformer =  feature_transformer

    num_states = x_bin*y_bin
    num_actions = 5
    self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

  def predict(self, observation):
    x = self.feature_transformer.transform(observation)
    return self.Q[x]

  def update(self, observation, a, G):
    x = self.feature_transformer.transform(observation)
    alpha = 0.01
    self.Q[x,a] += alpha*(G - self.Q[x,a])

  def act(self, observation, eps):
    if np.random.random() < eps:
      return np.random.choice(possible_action)
    else:   
      p = self.predict(observation)
      return np.argmax(p)
      
      
def state_that_contain_line(start_pos, goal_pos):
    m = (goal_pos[1]-start_pos[1])/(goal_pos[0]-start_pos[0])
    # y = mx 
    #discretize by 1
    discrete_x = range(start_pos[0], goal_pos[0]+1)
    discrete_y = []
    for x in discrete_x:
        discrete_y.append(int(m*x))
    
    ft = FeatureTransformer()
    state=[]
    for i in range(len(discrete_x )):
        if ft.transform_axis( (discrete_x[i],discrete_y[i])) not in state:
            state.append(ft.transform_axis( (discrete_x[i],discrete_y[i]) ))
    return state  #return list of state that contain line from start to end   
        

    
      
def step(observation, action):
    #next observation
    client.moveToPosition(observation.x_val+ interpret_action(action)[0]
    ,observation.y_val +interpret_action(action)[1]
    ,observation.z_val + interpret_action(action)[2] ,vel)
    
    new_observation = client.getPosition()
#This can be continous space
    FT = FeatureTransformer()
    line_state = state_that_contain_line(start_pos, goal_pos)
    #rewards
    if FT.transform(new_observation) in line_state:
        reward = 0
    else:
        reward = -2

    return new_observation, reward
    
def isDone(reward):
    done = 0
    if  reward <= -50:
        done = 1
    return done
    
def play_one(agent, eps, gamma,start_pos,vel):
    observation = client.getPosition()
    done = 0
    totalreward = 0
    iters = 0
    while not done and iters<20:
        action = agent.act(observation, eps)
        prev_observation = observation

        observation, reward = step(observation,action)
        
        
        totalreward += reward
        done = isDone(totalreward)
        
        if done:
            reward = -100
            
        #if reach terminal state
        if (observation.x_val<=goal_pos[0]+5 and observation.x_val>=goal_pos[0]-5) and  (observation.y_val<=goal_pos[1]+5 and observation.y_val>=goal_pos[1]-5):
            client.hover()
            time.sleep(1)
            reward = 300
            G = reward + gamma*np.max(agent.predict(observation))
            agent.update(prev_observation, action, G)
            break

    # update the model
        G = reward + gamma*np.max(agent.predict(observation))
        agent.update(prev_observation, action, G)

        iters += 1
        
    client.moveToPosition(start_pos[0],start_pos[1],start_pos[2],vel)

    return totalreward      
    
def plot_running_avg(totalrewards):
    N = len(totalrewards) 
    running_avg = np.empty(N)
    for t in range(N):
      running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()
  

#start
start_pos = [0,0,-18]
#goal
goal_pos = [12,101,-18]
#distance to goal
dist_to_goal = np.sqrt((start_pos[0]-goal_pos[0])**2 + (start_pos[1]-goal_pos[1])**2 )
#moving velocity
vel = 3

# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoff(1)


#go to start pos
client.moveToPosition(start_pos[0],start_pos[1],start_pos[2],vel)
#wait
client.hover()
time.sleep(1)

'''
#Adjust yaw
yaw_rate = 1
to_angle = 180*np.arctan((goal_pos[0]-start_pos[0])/(goal_pos[1]-start_pos[1]))/np.pi
client.rotateByYawRate(-np.sign((goal_pos[0]-start_pos[0])/(goal_pos[1]-start_pos[1]))*yaw_rate, duration=np.abs(to_angle/yaw_rate))
time.sleep(np.abs(to_angle/yaw_rate))
'''
ft = FeatureTransformer()
agent = Agent(ft)
gamma = 0.9

#training
N = 600
totalrewards = np.empty(N)
for n in range(N):
    eps = 1/np.sqrt(n+1)
    totalreward = play_one(agent, eps, gamma,start_pos,vel)
    totalrewards[n] = totalreward
    if n % 5 == 0:
        print("episode:", n, "total reward:", totalreward, "eps:", eps)
        print("avg reward for last 5 episodes:", totalrewards[-5:].mean())

Q_final = agent.Q        

plt.plot(totalrewards)
plt.title("Rewards")
plt.show()

plot_running_avg(totalrewards)