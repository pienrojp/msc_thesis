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


'
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


class Noise:
    def __init__(self, var, min_var, decay_of_var):
        self.decay_of_var
        self.var = var
        self.min_var = min_var
        
    def apply_noise(self, epis):
        norm = tf.contrib.distributions.Normal(0, var - np.max(min_var, epis*self.decay_of_var))
        return norm.sample()
        
    
# approximates pi(a | s)
class PolicyModel:
    def __init__(self, D, ft, hidden_layer_sizes=[]):
        self.ft = ft

    ##### hidden layers #####
        M1 = D
    
        self.hidden_layers = []

        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.hidden_layers.append(layer)
            M1 = M2

    # final layer mean
        self.output = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)

    # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
        
    # get final hidden layer
        Z = self.X
        for layer in self.hidden_layers:
            Z = layer.forward(Z)

    # calculate output and cost
        act = self.output.forward(Z)
    # make them 1-D
        act = tf.reshape(act, [-1])
        
        
        
        self.predict_op = act
        
        #log pi(a|s)
        log_probs = tf.log(self.actions)  # +noise ??
        
        
 #       norm = tf.contrib.distributions.Normal(mean, stdv)
 #       self.predict_op = norm.sample()
    
    #log pi(a|s)
 #       log_probs = norm.log_prob(self.actions)

   
        cost = -tf.reduce_sum(self.advantages * log_probs)
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

    def sample_action(self, X, epis):
        noise = Noise(var = 1, min_var= 0.1, decay_of_var = 0.02)
        p = self.predict(X)[0] + noise.apply_noise(epis)
        return p


# approximates V(s)
class ValueModel:
    def __init__(self, D, ft, hidden_layer_sizes=[]):
        self.ft = ft
        self.costs = []

    # create the graph
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

    # final layer
        layer = HiddenLayer(M1, 1, lambda x: x)
        self.layers.append(layer)

    # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

    # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
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
    epis=0

    while not done:
        action = pmodel.sample_action(observation, epis)
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
        epis += 1

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
