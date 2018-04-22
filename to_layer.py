from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

#from q_learning import plot_running_avg, FeatureTransformer, plot_cost_to_go


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
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))

    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer

  def transform(self, observations):
    # print "observations:", observations
    scaled = self.scaler.transform(observations)
    return self.featurizer.transform(scaled)


# so you can test different architectures
class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.relu, use_bias=True, zeros=False):
    if zeros:
      W = np.zeros((M1, M2), dtype=np.float32)
    else:
      W = tf.random_normal(shape=(M1, M2)) * np.sqrt(2. / M1, dtype=np.float32)
    self.W = tf.Variable(W)

    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))

    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)


# approximates pi(a | s)
class PolicyModel:
    def __init__(self, D, ft, hidden_layer_sizes=[]):
        self.ft = ft
    
    # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
            
        #### Architecture ####
        
    #hidden layer
        nn = tf.layers.dense(self.X, 100, activation=tf.nn.tanh)
        nn = tf.layers.dense(nn, 50, activation=tf.nn.tanh)
        
    # final layer mean
        self.mean_layer = tf.layers.dense(nn, 1)  #linear activation

    # final layer variance
        self.stdv_layer = tf.layers.dense(nn, 1, activation=tf.nn.softplus)

        #####################

        #OR
        mean = tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT, predictions=self.mean_layer)
        stdv = tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT, predictions=self.stdv_layer)

    # calculate output and cost
        #mean = self.mean_layer.predict(self.X)
        #stdv = self.stdv_layer.predict(self.X)
        
    # make them 1-D
        mean = tf.reshape(mean, [-1])
        stdv = tf.reshape(stdv, [-1]) 

        norm = tf.contrib.distributions.Normal(mean, stdv)
        self.predict_op = norm.sample()
    
    #log pi(a|s)
        log_probs = norm.log_prob(self.actions)
    
        cost = -tf.reduce_sum(self.advantages * log_probs + 0.1*norm.entropy())
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
    
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.session.run(self.train_op,feed_dict={self.X: X,self.actions: actions,self.advantages: advantages})

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        return p


# approximates V(s)
class ValueModel:
    def __init__(self, D, ft, hidden_layer_sizes=[]):
        self.ft = ft
        self.costs = []

    # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')
        
        
    #hidden layer
        nn = tf.layers.dense(self.X, 100, activation=tf.nn.tanh)
        nn = tf.layers.dense(nn, 50, activation=tf.nn.tanh)
        
    # final layer mean
        self.Y_FL = tf.layers.dense(nn, 1)  #linear activation
        
    # calculate output and cost
        Z = tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT, predictions=self.Y_FL)
         
        Y_hat = tf.reshape(Z, [-1]) # the output
    
        self.predict_op = Y_hat

        cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.cost = cost
        self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        Y = np.atleast_1d(Y)
    #fit one step
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
    
        cost = self.session.run(self.cost, feed_dict={self.X: X, self.Y: Y})
        self.costs.append(cost)

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})


    
def step(observation, action, timestep=1, K=1):
    #next observation
    client.moveByVelocity(v_x, v_y, action ,timestep)
    time.sleep(0.8)
    
    #like depth from depth camera 
    new_observation = client.getPosition().z_val - z_wire
    #rewards
    reward = -K*(new_observation-z_dist_wanted)**2

    return new_observation, reward    
    
    

def isDone(reward):
    done = 0
    if  reward <= -25:
        done = 1
    return done    

def isDonebyCollide(reward):
    done_col = 0
    collision_info = client.getCollisionInfo()
    if collision_info.has_collided:
        done_col =1 
    return done_col  

    
    
def play_one_td(pmodel, vmodel, gamma):
    observation = client.getPosition().z_val - z_wire
    done = 0
    done_col=0
    totalreward = 0

    while not done:
        action = pmodel.sample_action(observation)
        prev_observation = observation
        #1 step action
        observation, reward = step(observation,action)
        
        done = isDone(totalreward)
    
        if done:
            reward = -100
            
        if done_col:
            reward = -1000
            
        #if reach terminal state
        if (observation.x_val<=goal_pos[0]+1 and observation.x_val>=goal_pos[0]-1) and  (observation.y_val<=goal_pos[1]+1 and observation.y_val>=goal_pos[1]-1):
            client.hover()
            time.sleep(1)
            break
    
    
        totalreward += reward

        V_next = vmodel.predict(observation)
    
        G = reward + gamma*V_next
        advantage = G - vmodel.predict(prev_observation)
        pmodel.partial_fit(prev_observation, action, advantage)
        vmodel.partial_fit(prev_observation, G)

    client.moveToPosition(start_pos[0],start_pos[1],start_pos[2],vel)
    return totalreward

    
    
    
''' Main '''

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
D = ft.dimensions
pmodel = PolicyModel(D, ft, [1024,1024])
vmodel = ValueModel(D, ft, [1024,1024])
init = tf.global_variables_initializer()
session = tf.InteractiveSession()
session.run(init)
pmodel.set_session(session)
vmodel.set_session(session)
gamma = 0.9
N = 50
totalrewards = np.empty(N)
costs = np.empty(N)
#play N times
for n in range(N):
    totalreward, num_steps = play_one_td(pmodel, vmodel, gamma)
    totalrewards[n] = totalreward
    print("episode:", n, "total reward: %.1f" % totalreward)
    
print("avg reward for last 100 episodes:", totalrewards[-100:].mean())

plt.plot(totalrewards)
plt.title("Rewards")
plt.show()
