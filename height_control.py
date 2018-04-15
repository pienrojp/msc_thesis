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
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

import sys
sys.path.insert(0, '/home/pienrojp/AirSim/PythonClient')
from AirSimClient import *


class SGDRegressor:
  def __init__(self, D):
    self.w = np.random.randn(D) / np.sqrt(D)
    self.lr = 0.1

  def partial_fit(self, X, Y):
    self.w += self.lr*(Y - X.dot(self.w)).dot(X)

  def predict(self, X):
    return X.dot(self.w)




''' start position // goal position'''

#start
start_pos = [0,0,-18]
#goal
goal_pos = [12,101,-18]
#distance to goal
dist_to_goal = np.sqrt((start_pos[0]-goal_pos[0])**2 + (start_pos[1]-goal_pos[1])**2 )
#moving velocity
vel = 3
#slope
m = (goal_pos[1]-start_pos[1])/(goal_pos[0]-start_pos[0])
#vx and vy
v_y = vel
v_x = vel/m

#desired_height
z_wire = -15
z_dist_wanted = 2

''' state = (x,y,z) and V(x,y,z) ~ f(x,y,z) where f() is rbf net'''
class FeatureTransformer:
  def __init__(self, n_components=500):
      #lets say [8,-8]
    observation_examples = 8*(np.random.random((20000, 1))*2 - 1)
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=n_components))
            ])
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))

    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer

  def transform(self, observations):
    # print "observations:", observations
    scaled = self.scaler.transform(observations)
    return self.featurizer.transform(scaled)


''' Action '''
possible_action = [-1, -0.7, -0.5, -0.3, 0, 0.3, 0.5, 0.7, 1]

#5 discrete actions
def interpret_action(action):
    quad_offset = (0, 0, 0)
    
    if action == -1:
        quad_offset = (0, 0, -1)
    elif action == -0.7:
        quad_offset = (0, 0, -0.7)
    elif action == -0.5:
        quad_offset = (0, 0, -0.5)
    elif action == -0.3:
        quad_offset = (0, 0, -0.3)
    elif action == 0:
        quad_offset = (0, 0, 0)
    elif action == 0.3:
        quad_offset = (0, 0, 0.3)
    elif action == 0.5:
        quad_offset = (0, 0, 0.5)
    elif action == 0.7:
        quad_offset = (0, 0, 0.7)
    elif action == 1:
        quad_offset = (0, 0, 1)
        
        
    return quad_offset[2]
    
    
''' Control '''

class Agent:
  def __init__(self, feature_transformer):
    self.feature_transformer =  feature_transformer
    self.models = []
    
    #array of model same number as action
    for i in range(len(possible_action)):
        model = SGDRegressor(feature_transformer.dimensions)
        self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    result = np.stack([m.predict(X) for m in self.models]).T
    return result

  def update(self, observation, a, G):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    self.models[a].partial_fit(X, [G])

  def act(self, observation, eps):
    if np.random.random() < eps:
      return np.random.choice(possible_action)
    else:   
      p = self.predict(observation)
      return np.argmax(p)
      

    
      
def step(observation, action, K=1):
    #next observation
    client.moveByVelocity(v_x, v_y, interpret_action(action) ,time_step)
    time.sleep(0.5)
    
    #like depth from depth camera 
    new_observation = client.getPosition().z_val - z_wire
    #rewards
    reward = -K*(new_observation-z_dist_wanted)**2

    return new_observation, reward
    
def isDone(reward):
    done = 0
    collision_info = client.getCollisionInfo()
    
    if  reward <= -25:
        done = 1
        
    elif collision_info.has_collided:
        done =1 
        
    return done
    
def play_one(agent, eps, gamma,start_pos,vel):
    observation = client.getPosition().z_val - z_wire
    done = 0
    totalreward = 0
    
    while not done:
        action = agent.act(observation, eps)
        prev_observation = observation

        observation, reward = step(observation,action)
        
        
        totalreward += reward
        done = isDone(totalreward)
        
        if done:
            reward = -100
            
        #if reach terminal state
        if (observation.x_val<=goal_pos[0]+1 and observation.x_val>=goal_pos[0]-1) and  (observation.y_val<=goal_pos[1]+1 and observation.y_val>=goal_pos[1]-1):
            client.hover()
            time.sleep(1)
            break

    # update the model
        #[0] ??
        G = reward + gamma*np.max(agent.predict(observation))
        agent.update(prev_observation, action, G)

        
    #move back to init
    client.moveToPosition(start_pos[0],start_pos[1],start_pos[2],vel)

    return totalreward      
    
    

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