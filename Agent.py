#Imports
#from DQN import DQN #Change According to NN naming
#import torch
#import torch.nn.functional as F
import numpy as np


class Agent():
    def __init__(self, memory_capacity=0, batch_size=0, observation_space=0, action_space=0, hidden_neurons=0, learning_rate=0, epsilon=0.3, gamma=0.95) -> None:
        #Neural Network Training Device
        #self.device = torch.device("cpu")
        
        #Experience Replay
        #Put Replay Object here
        
        #Policy Network
        #self.network = 
        #self.target_network = 
        #self.target_network.load_state_dict(self.network.state_dict())
        #self.target_network.eval()
        #self.network.eval()
        
        #Network Optimiser - Have a look at different optimisers
        #The optimiser updates the weightings in the nn to predict better q-values.
        #self.optimiser = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        #Hyperparameters
        self.epsilon = epsilon
        self.gamma = gamma
        
    def action(self, state):
        # if torch.rand(1) < self.epsilon:
        #     return torch.randint(0, self.network.output_size, (1,))
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
        
    
    def episode(self, env):
        state, info = env.reset()
        total_reward = 0
        terminated = False
        while not terminated:
            action = self.action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            #Put Experience Replay Here
            state = next_state
            total_reward += reward
        #self.update_target()
        return total_reward
    
    def train(self, env, episodes):
        #Add in performance indicators
        rewards = []
        for episode in range(episodes):
            reward = self.episode(env)
            rewards.append(reward)
        return rewards
    