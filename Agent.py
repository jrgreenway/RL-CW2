#Imports
#from DQN import DQN #Change According to NN naming
import torch
#import torch.nn.functional as F
import numpy as np


class Agent():
    def __init__(self, observation_size, action_size, epsilon=0.3, gamma=0.95, **kwargs):
        #Neural Network Training Device
        #self.device = torch.device("cpu")
        
        #Env Infomation
        self.observation_size = observation_size
        self.action_size = action_size
        
        #Experience Replay
        #Put Replay Object here
        
        #Policy Network - If we want a target network
        #self.network = 
        #self.target_network = 
        
        
        #Network Optimiser
        #self.optimiser = 
        
        #Hyperparameters
        self.epsilon = epsilon
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.99)
        self.gamma = gamma
        
    def action(self, state):
        # if torch.rand(1) < self.epsilon:
        #     return torch.randint(0, self.action_size, (1,))
        # else:
        #     with torch.no_grad():
        #         action = self.network(state).argmax() #Update according to nn parameters
        action = np.random.randint(0,5)
        return action
    
    def learn_replay(self):
        #Insert Experience replay learning here
        pass
    
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def decay(self):
        self.epsilon = max(self.epsilon_minimum ,self.epsilon * self.epsilon_decay)
    
    def episode(self, env):
        state, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = self.action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            #Put Experience Replay Here
            state = next_state
            total_reward += reward
        return total_reward
    
    def train(self, env, episodes):
        #Add in performance indicators
        rewards = []
        for episode in range(episodes):
            reward = self.episode(env)
            rewards.append(reward)
        return rewards
    