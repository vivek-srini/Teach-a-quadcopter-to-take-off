# TODO: your agent here!

from keras import layers, models, optimizers, regularizers
from keras import backend as K

import numpy as np
from task import Task

import random
from collections import namedtuple, deque

class ReplayBuffer:                                 #This is the same implementation of replay buffer provided in the udacity lesson
    

    def __init__(self, buffer_size, batch_size):
       
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        
        return len(self.memory)
    
class Actor():
    
    def __init__(self, state_size, action_size, action_low, action_high):
        
        self.state_size = state_size 
        self.action_size = action_size 
        self.action_low = action_low
        self.action_high = action_high 
        self.action_range = self.action_high - self.action_low
        
        self.build_model()
        
    def build_model(self):
        
        states = layers.Input(shape=(self.state_size ,), name = 'states')             #Input Layer
        
        
        net=layers.Dense(units=32,use_bias=True,kernel_regularizer=regularizers.l2(0.02))(states)         #First hidden layer
        net=layers.BatchNormalization()(net)
        net=layers.Activation('relu')(net)
        net=layers.Dropout(0.4)(net)
        
        net=layers.Dense(units=250,use_bias=True,kernel_regularizer=regularizers.l2(0.02))(net)          #Second hidden layer
        net=layers.BatchNormalization()(net)
        net=layers.Activation('relu')(net)
        net=layers.Dropout(0.4)(net)
        
        net=layers.Dense(units=350,use_bias=True,kernel_regularizer=regularizers.l2(0.02))(net)          #Third hidden layer
        net=layers.BatchNormalization()(net)
        net=layers.Activation('relu')(net)
        net=layers.Dropout(0.5)(net)
        
        
        
        raw_actions = layers.Dense(units = self.action_size, activation = 'sigmoid', name = 'raw_actions')(net)   #Output layer
    
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name = 'actions')(raw_actions)
        
        self.model = models.Model(inputs = states, outputs = actions)
    
       
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)
        optimizer = optimizers.Adam(0.01)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], outputs=[], updates=updates_op)
        
class Critic():
    
    def __init__(self, state_size, action_size):
        
        
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()
        
    def build_model(self):
        
        states = layers.Input(shape=(self.state_size,), name='states')          #Input layers for state and actions
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        
        net_states=layers.Dense(units=32,use_bias=True,kernel_regularizer=regularizers.l2(0.01))(states)   #Hidden layer 1 for states
        net_states=layers.BatchNormalization()(net_states)
        net_states=layers.Activation('relu')(net_states)
        net_states=layers.Dropout(0.5)(net_states)
        
        net_states=layers.Dense(units=300,use_bias=True,kernel_regularizer=regularizers.l2(0.01))(net_states)  #Hidden layer 2 for states
        net_states=layers.BatchNormalization()(net_states)
        net_states=layers.Activation('relu')(net_states)
        net_states=layers.Dropout(0.5)(net_states)
        
        
        
        
        net_actions=layers.Dense(units=32,use_bias=True,kernel_regularizer=regularizers.l2(0.01))(actions)   #Hidden layer 1 for actions
        net_actions=layers.BatchNormalization()(net_actions)
        net_actions=layers.Activation('relu')(net_actions)
        net_actions=layers.Dropout(0.5)(net_actions)
        
        net_actions=layers.Dense(units=300,use_bias=True,kernel_regularizer=regularizers.l2(0.01))(net_actions)  #Hidden layer 2 for actions
        net_actions=layers.BatchNormalization()(net_actions)
        net_actions=layers.Activation('relu')(net_actions)
        net_actions=layers.Dropout(0.5)(net_actions)
        
       
        
        
        
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
        
        Q_values = layers.Dense(units=1, name='q_values')(net)
        
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        
        optimizer = optimizers.Adam(0.01)
        self.model.compile(optimizer=optimizer, loss='mse')
        
        action_gradients = K.gradients(Q_values, actions)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()],outputs=action_gradients)
        
class DDPG():
    
    def __init__(self, Task):
        self.task = Task
        self.state_size = Task.state_size
        self.action_size = Task.action_size
        self.action_low = Task.action_low
        self.action_high = Task.action_high

        
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         
        self.memory.add(self.last_state, action, reward, next_state, done)

        
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        
        self.last_state = next_state

    def act(self, states):
        
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  

    def learn(self, experiences):
        
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

       
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  

        
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
class OUNoise:
    
    def __init__(self, size, mu, theta, sigma):
        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
       
        self.state = self.mu

    def sample(self):
        
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
    
        
        