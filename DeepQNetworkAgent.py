#Imports
#from DQN import DQN #Change According to NN naming
import gymnasium
import torch as T
import torch.nn as nn               # Neural Network (nn)
# We may need to import something for convulutional layers because we are working with images
import torch.nn.functional as F     # Activation Functions
import torch.optim as optim         # nn Optimiser
import numpy as np                  # numpy
from termcolor import colored       # Colored text for debugging
import os

from tqdm import tqdm                           # For file joining operations to handle model checkpointing

# This inmplementation has no target network or convulutional layers
# Convulutional layers are used to process the image and extract features from it
# Target network is basically a copy of the main network that is updated every few steps


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
        #Returns a batch of random experiences of size batch_size
        batch_indexes = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(states=self.state_memory[batch_indexes], 
                    next_states=self.next_state_memory[batch_indexes],
                    actions=self.action_memory[batch_indexes],
                    rewards=self.reward_memory[batch_indexes],
                    dones=self.done_memory[batch_indexes])

    def __len__(self):
        return self.size

class DuelingDeepQNetwork(nn.Module):
    # lr            = learning rate
    # input_dims    = input dimensions
    # fc1_dims      = fully connected layer 1 dimensions
    # fc2_dims      = fully connected layer 2 dimensions
    # n_actions     = number of actions
    def __init__(self, lr, input_dims, nn_dims, n_actions, checkpoint_dir, name):
        super(DuelingDeepQNetwork, self).__init__()    # Inheriting from nn.Module

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)  # Save our checkpoint directory for later use

        self.feature = nn.Sequential(   #Feature layer that leads into advantage and value streams
            nn.Linear(input_dims, nn_dims),
            nn.ReLU(),
        )
        
        self.V = nn.Sequential( # Value stream: tells the agent the value of the current state
            nn.Linear(nn_dims, nn_dims),
            nn.ReLU(),
            nn.Linear(nn_dims, 1),
        )
        
        self.A = nn.Sequential( # Advantage stream: tells the agent the relative advantage of eachh action in a given state
            nn.Linear(nn_dims, nn_dims),
            nn.ReLU(),
            nn.Linear(nn_dims, n_actions)
        )     

    def forward(self, state):
        x = self.feature(state)
        V = self.V(x)
        A = self.A(x)
        return V + A - A.mean(dim=-1, keepdim=True)
    
    #Save and load checkpoints
    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, file):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
    


class DQNAgent():
    def __init__(self, env: gymnasium.Env, learning_rate, batch_size, gamma, epsilon, max_memory_size=100000, hidden_neurons=64, eps_decay=0.9995, eps_min=0.1, replace=1000, checkpoint_dir='tmp/'): 
        # Adjust epsilon decay rate later, right now linear decay

        self.env = env
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.observation_space = env.observation_space
        self.observation_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.action_space = [i for i in range(self.n_actions)]
        self.memory_size = max_memory_size
        
        self.memory = ReplayMemory(self.observation_shape[0], max_memory_size, batch_size)
        
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma

        self.replace_target_count = replace
        self.checkpoint_dir = checkpoint_dir

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        # Q Evaluation Network
        self.network = DuelingDeepQNetwork(self.learning_rate, input_dims=self.observation_shape[0], nn_dims=hidden_neurons,
                                    n_actions=self.n_actions, name='surround_dueling_ddqn',
                                    checkpoint_dir=self.checkpoint_dir)
        
        self.target_network = DuelingDeepQNetwork(self.learning_rate, input_dims=self.observation_shape[0], nn_dims=hidden_neurons,
                                    n_actions=self.n_actions, name='surround_dueling_ddqn_target',
                                    checkpoint_dir=self.checkpoint_dir)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        
        self.optimiser = optim.Adam(self.network.parameters())
        
        self.transition = []
        
        self.testing = False
         
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
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()

    def learn(self):
        batches = self.memory.sample()
        loss = self.calculate_loss(batches) # make helper function
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1) # clips gradients between -1 and 1, can change
        self.optimiser.step()
        return loss.item()

    def calculate_loss(self, samples):
        """Helper function to simplify learn() function, returns the loss"""
        state = T.FloatTensor(samples["states"]).to(self.device)
        next_state = T.FloatTensor(samples["next_states"]).to(self.device)
        action = T.LongTensor(samples["actions"].reshape(-1, 1)).to(self.device)
        reward = T.FloatTensor(samples["rewards"].reshape(-1, 1)).to(self.device)
        done = T.FloatTensor(samples["dones"].reshape(-1, 1)).to(self.device)

        current_q = self.network(state).gather(1, action)
        next_q = self.target_network(next_state).max(dim=1, keepdim=True)[0].detach()
        if_done = 1 - done # sets gamma*Q to zero if step lead to termination or truncation
        target = (reward + self.gamma * next_q * if_done).to(self.device)

        loss = F.smooth_l1_loss(current_q, target) # Currently on smooth_l1_loss, can change to MSE or others.

        return loss

    def decay_epsilon(self, linear=False):
        if linear:
            self.epsilon = max(self.eps_min, self.epsilon - self.eps_decay)
        else:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        
    def train(self, episodes):
        self.testing = False
        tracked_info = {
            "scores":[],
            "losses":[],
            "epsilons":[]
        }
        learn_count = 0
        
        state, _ = self.env.reset()
        for episode in tqdm(range(1,episodes+1)):
            score = 0
            while True:
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                state = next_state
                done = terminated or truncated
                # if not done:
                #     reward += 0.01
                score += reward
                
                if not self.testing:
                    self.transition += [reward, next_state, done]
                    self.memory.store_transition(*self.transition)
                
                if done:
                    state, _ = self.env.reset()
                    tracked_info["scores"].append(score)
                    score = 0
                    if episode > 10 and episode % 10 == 0:
                        self.save_models()
                    break
                
                if len(self.memory) >= self.batch_size:
                    loss = self.learn()
                    tracked_info["losses"].append(loss)
                    learn_count += 1
                    self.decay_epsilon(linear=False)
                    tracked_info["epsilons"].append(self.epsilon)
                    self.replace_target_network(learn_count)
                
        self.env.close()
        return tracked_info
                
            

