#Imports
from collections import deque
from re import L
import gymnasium
import torch as T
import torch.nn as nn               # Neural Network (nn)
import torch.nn.functional as F     # Activation Functions
import torch.optim as optim         # nn Optimiser
import numpy as np                  # numpy
import random
import os
import math
from segment_tree import SumSegmentTree, MinSegmentTree
from tqdm import tqdm
import logging

class ReplayMemory():
    def __init__(self, observation_dims, memory_size, batch_size):
        # Experience memory
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_index = 0
        self.size = 0
        self.state_memory = np.zeros((memory_size, observation_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((memory_size, observation_dims), dtype=np.float32)
        self.action_memory = np.zeros((memory_size), dtype=np.int32)
        self.reward_memory = np.zeros((memory_size), dtype=np.float32)
        self.done_memory = np.zeros(memory_size, dtype=bool)
    
    def store_transition(self, state, action, reward, next_state, done):
        #Store transitions in memory so that we can use them for experience replay   
        self.state_memory[self.memory_index] = state
        self.action_memory[self.memory_index] = action
        self.reward_memory[self.memory_index] = reward
        self.next_state_memory[self.memory_index] = next_state
        self.done_memory[self.memory_index] = done
        self.memory_index = (self.memory_index + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)
        
    def sample(self):
        #Sample a batch of transitions from memory
        max_memory = min(self.size, self.memory_size)
        batch = np.random.choice(max_memory, self.batch_size, replace=False)
        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.done_memory[batch]
        return {"states":states, "actions":actions, "rewards":rewards, "next_states":next_states, "dones":dones}

    def __len__(self):
        return self.size
        
class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, nn_dims, checkpoint_dir, name):
        super(DeepQNetwork, self).__init__()    # Inheriting from nn.Module

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)  # Save our checkpoint directory for later use

        self.feature = nn.Sequential(
            nn.Linear(input_dims, nn_dims),
            nn.ReLU(),
            nn.Linear(nn_dims, nn_dims),
            nn.ReLU(),
            nn.Linear(nn_dims, output_dims)
        )

    def forward(self, x: T.tensor):
        feat = self.feature(x)
        return feat
    
    #Save and load checkpoints
    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, file):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
    


class DQNAgent():
    def __init__(self, env: gymnasium.Env, learning_rate, batch_size, gamma, epsilon, eps_min, max_memory_size=100000, hidden_neurons=128, replace=1000, checkpoint_dir='tmp/', log=True): 
        # Adjust epsilon decay rate later, right now linear decay

        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.eps_max = epsilon
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.observation_space = env.observation_space
        self.observation_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.action_space = [i for i in range(self.n_actions)]
        self.memory_size = max_memory_size
        
        self.memory = ReplayMemory(self.observation_shape[0], max_memory_size, batch_size)
        
        self.gamma = gamma

        self.replace_target_count = replace
        self.checkpoint_dir = checkpoint_dir

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        print(self.device.type)

        # Q Evaluation Network
        self.network = DeepQNetwork(input_dims=self.observation_shape[0], output_dims=self.n_actions, nn_dims=hidden_neurons,
                                    name='surround_dueling_ddqn',
                                    checkpoint_dir=self.checkpoint_dir)
        
        self.target_network = DeepQNetwork(input_dims=self.observation_shape[0], output_dims=self.n_actions, nn_dims=hidden_neurons,
                                    name='surround_dueling_ddqn_target',
                                    checkpoint_dir=self.checkpoint_dir)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        
        self.optimiser = optim.Adam(self.network.parameters())
        
        self.transition = []
        
        self.testing = False
        
        if log:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)
            self.handler = logging.FileHandler('logs.log')
            self.handler.setLevel(logging.INFO)
            self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            self.handler.setFormatter(self.formatter)
            self.logger.addHandler(self.handler)
            
         
    def action(self, state: np.ndarray):
        '''Action choice based on epsilon-greedy policy, returns the action as a np.ndarray''' 
        if np.random.random() > self.epsilon:
            action = self.network(T.FloatTensor(state).to(self.device)).argmax()
            action = action.detach().cpu().numpy()
        else:
            action = np.random.choice(self.action_space)
        if not self.testing:
            self.transition = [state, action]
        return action
    
    def replace_target_network(self, count):
        '''Updates target network on regular intervals'''
        if count % self.replace_target_count == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            
    def save_models(self):
        self.network.save_checkpoint()
        self.target_network.save_checkpoint()

    def load_models(self):
        self.network.load_checkpoint()
        self.target_network.load_checkpoint()

    def learn(self):
        batches = self.memory.sample()
        loss = self.calculate_loss(batches)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return loss.item()

    def calculate_loss(self, samples):
        """loss function to simplify learn() function"""
        state = T.FloatTensor(samples["states"]).to(self.device)
        next_state = T.FloatTensor(samples["next_states"]).to(self.device)
        action = T.LongTensor(samples["actions"].reshape(-1,1)).to(self.device)
        reward = T.FloatTensor(samples["rewards"].reshape(-1, 1)).to(self.device)
        done = T.FloatTensor(samples["dones"].reshape(-1, 1)).to(self.device)
        current_q = self.network(state).gather(1, action)
        next_q = self.target_network(next_state).max(dim=1, keepdim=True)[0].detach()
        target_q = (reward+self.gamma*next_q*(1-done)).to(self.device)
        loss = F.smooth_l1_loss(current_q, target_q)
        return loss

    def decay_epsilon(self, step, steps, linear=False):
        if step >= steps: return
        if linear:
            dec = (self.eps_max - self.eps_min) / steps
            self.epsilon = max(self.eps_min, self.eps_max - dec * step)
        else:
            self.epsilon = self.eps_min + (self.eps_max - self.eps_min) * math.exp(-1. * step * math.log((self.eps_max / self.eps_min), math.e) / steps)
        
    def make_params_dict(self):
        return dict(
            env=self.env.spec.id,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gamma=self.gamma,
            eps_max=self.eps_max,
            eps_min=self.eps_min,
            max_memory_size=self.memory_size,
            replace=self.replace_target_count,
        )
        
    
    def train(self, steps):
        self.testing = False
        tracked_info = {
            "scores":[],
            "losses":[],
            "epsilons":[]
        }
        parameters = self.make_params_dict()
        learn_count = 0
        episodes = 1
        score = 0
        state, _ = self.env.reset()
        for step in tqdm(range(1,steps+1)):
            action = self.action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            state = next_state
            done = terminated or truncated
            score += reward
            
            if not self.testing:
                self.transition += [reward, next_state, done]
                self.memory.store_transition(*self.transition)
            
            if done:
                state, _ = self.env.reset()
                tracked_info["scores"].append(score)
                tracked_info["epsilons"].append(self.epsilon)
                self.logger.info(f"Ep. Num.: {episodes}, Ep. Score: {score}, Avg. Score: {np.mean(tracked_info['scores'][-10:])}")
                score = 0
                episodes+=1
                if episodes > 10 and episodes % 10 == 0:
                    self.save_models()
            
            if len(self.memory) >= self.batch_size:
                loss = self.learn()
                tracked_info["losses"].append(loss)
                self.decay_epsilon(step, steps*0.25)
                learn_count += 1
                self.replace_target_network(learn_count)
                
        self.env.close()
        self.save_models()
        return tracked_info, parameters
                
            

