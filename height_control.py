# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:24:22 2018

@author: pienrojp
"""
from __future__ import print_function, division
from builtins import range
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pylab as plt

import sys
sys.path.insert(0, '/home/pienrojp/AirSim/PythonClient')
from AirSimClient import *


class SGDRegressor:
  def __init__(self, D):
    self.w = np.random.randn(D) / np.sqrt(D)
    self.lr = 0.02

  def partial_fit(self, X, Y):
    self.w += self.lr*(Y - X.dot(self.w)).dot(X)

  def predict(self, X):
    return X.dot(self.w)

class SGDRegressor_cont:
  def __init__(self, w):
    self.w = w
    self.lr = 0.02

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
vel = 2
#slope
m = (goal_pos[1]-start_pos[1])/(goal_pos[0]-start_pos[0])
#vx and vy
v_y = vel
v_x = vel/m

#desired_height
z_wire = -15
z_dist_wanted = -2

''' state = (x,y,z) and V(x,y,z) ~ f(x,y,z) where f() is rbf net'''
class FeatureTransformer:
  def __init__(self, n_components=100):
      #lets say [8,-8]
    observation_examples = 11*(np.random.random((20000, 1)) - 0.5)
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
possible_action = [0, 1, 2, 3, 4, 5, 6,7,8]

#5 discrete actions
def interpret_action(action):
    quad_offset = (0, 0, 0)
    
    if action == 0:
        quad_offset = (0, 0, -2)
    elif action == 1:
        quad_offset = (0, 0, -1)
    elif action == 2:
        quad_offset = (0, 0, -0.5)
    elif action == 3:
        quad_offset = (0, 0, 0)
    elif action == 4:
        quad_offset = (0, 0, 0.5)
    elif action == 5:
        quad_offset = (0, 0, 1)
    elif action == 6:
        quad_offset = (0, 0, 2)
    elif action == 7:
        quad_offset = (0, 0, -0.25)
    elif action == 8:
        quad_offset = (0, 0, 0.25)        
        
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
    X = self.feature_transformer.transform(np.atleast_2d(observation))
    self.models[a].partial_fit(X, [G])

  def act(self, observation, eps):
    if np.random.random() < eps:
      return np.random.choice(possible_action)
    else:   
      p = self.predict(observation)
      return np.argmax(p)
      
class Agent_cont:
  def __init__(self, feature_transformer,w_models_final):
    self.feature_transformer =  feature_transformer
    self.models = []
    
    #array of model same number as action
    for i in range(len(possible_action)):
        model = SGDRegressor_cont(w_models_final[i])
        self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    result = np.stack([m.predict(X) for m in self.models]).T
    return result

  def update(self, observation, a, G):
    X = self.feature_transformer.transform(np.atleast_2d(observation))
    self.models[a].partial_fit(X, [G])

  def act(self, observation, eps):
    if np.random.random() < eps:
      return np.random.choice(possible_action)
    else:   
      p = self.predict(observation)
      return np.argmax(p)
    
      
def step(observation, action, K=1):
    #next observation
    client.moveByVelocity(v_x, v_y, interpret_action(action) ,1)
    time.sleep(1)
    
    #like depth from depth camera 
    new_observation = client.getPosition().z_val - z_wire
    #rewards
    reward = -K*(new_observation-z_dist_wanted)**2

    return new_observation, reward
    
def isDone(reward):
    done = 0
    
    if  reward <= -30:
        done = 1
        
    return done
    
def done_by_collide():
    collision_info = client.getCollisionInfo()
    done_col = 0
    if collision_info.has_collided:
        done_col=1
    return done_col
    
def play_one(agent, eps, gamma,start_pos,vel):
    observation = client.getPosition().z_val - z_wire
    done = 0
    done_col = 0    
    totalreward = 0
    
    while not done:
        action = agent.act(observation, eps)
        prev_observation = observation

        observation, reward = step(observation,action)
        
        
        totalreward += reward
        done = isDone(totalreward)
        
        if done:
            reward = -100
            
        if done_col:
            reward = -1000
            G = reward + gamma*np.max(agent.predict(observation))
            agent.update(prev_observation, action, G)
            client.moveToPosition(start_pos[0],start_pos[1],start_pos[2],2*vel)
            return totalreward
            
            
        #if reach terminal state
        if (client.getPosition().x_val<=goal_pos[0]+1 and client.getPosition().x_val>=goal_pos[0]-1) and  (client.getPosition().y_val<=goal_pos[1]+1 and client.getPosition().y_val>=goal_pos[1]-1):
            client.hover()
            time.sleep(1)
            G = reward + gamma*np.max(agent.predict(observation))
            agent.update(prev_observation, action, G)
            client.moveToPosition(start_pos[0],start_pos[1],start_pos[2],2*vel)
            return totalreward  

    # update the model
        G = reward + gamma*np.max(agent.predict(observation))
        agent.update(prev_observation, action, G)

        
    #move back to init
    client.moveToPosition(start_pos[0],start_pos[1],start_pos[2],2*vel)
    client.hover()
    time.sleep(1.5)

    return totalreward      
    
    
models_final= np.load('/home/pienrojp/ms_thesis/result/height_Q_contS_disA/model.npy')
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


#Adjust yaw
yaw_rate = 1
to_angle = 180*np.arctan((goal_pos[0]-start_pos[0])/(goal_pos[1]-start_pos[1]))/np.pi
client.rotateByYawRate(-np.sign((goal_pos[0]-start_pos[0])/(goal_pos[1]-start_pos[1]))*yaw_rate, duration=np.abs(to_angle/yaw_rate))
time.sleep(np.abs(to_angle/yaw_rate))


ft = FeatureTransformer()
agent = Agent(ft)
#agent = Agent_cont(ft, models_final)
gamma = 0.9

#training
N = 60
totalrewards = np.empty(N)
for n in range(N):
    eps = 1/(n+1)
    totalreward = play_one(agent, eps, gamma,start_pos,vel)
    totalrewards[n] = totalreward
    print("episode:", n, "total reward:", totalreward, "eps:", eps)

client.armDisarm(False)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)

models_final = agent.models
w_models_final = []
for i in range(len(agent.models)):
    w_models_final.append(agent.models[i].w)


plt.plot(totalrewards)
plt.title("Rewards")
plt.xlabel('episodes')
plt.ylabel('rewards')
plt.show()


np.save('./result/height_Q_contS_disA/totalrewards', totalrewards)
np.save('./result/height_Q_contS_disA/model', models_final)
np.save('./result/height_Q_contS_disA/w_model', w_models_final)



#replay
models_final= np.load('/home/pienrojp/ms_thesis/result/height_Q_contS_disA/model.npy')
w_models_final = np.load('/home/pienrojp/ms_thesis/result/height_Q_contS_disA/w_model.npy')


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


#Adjust yaw
yaw_rate = 1
to_angle = 180*np.arctan((goal_pos[0]-start_pos[0])/(goal_pos[1]-start_pos[1]))/np.pi
client.rotateByYawRate(-np.sign((goal_pos[0]-start_pos[0])/(goal_pos[1]-start_pos[1]))*yaw_rate, duration=np.abs(to_angle/yaw_rate))
time.sleep(np.abs(to_angle/yaw_rate))


ft = FeatureTransformer()
agent = Agent_cont(ft, w_models_final)


gamma = 0.9

#training
N = 1
totalrewards = np.empty(N)
for n in range(N):
    eps = 0
    totalreward = play_one(agent, eps, gamma,start_pos,vel)
    totalrewards[n] = totalreward
    print("episode:", n, "total reward:", totalreward, "eps:", eps)

client.armDisarm(False)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)