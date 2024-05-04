#Imports
from collections import deque
import gymnasium
import torch as T
import torch.nn as nn               # Neural Network (nn)
# We may need to import something for convulutional layers because we are working with images
import torch.nn.functional as F     # Activation Functions
import torch.optim as optim         # nn Optimiser
import numpy as np                  # numpy
from termcolor import colored       # Colored text for debugging
import random
import os
import math
from segment_tree import SumSegmentTree, MinSegmentTree

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

    def __len__(self):
        return self.size

class PrioritisedReplay(ReplayMemory):
    '''Prioritised Experience Replay, alpha must be positive >=0.'''
    def __init__(self, observation_dims, memory_size, batch_size, alpha:float=0.6):
        super(PrioritisedReplay, self).__init__(observation_dims, memory_size, batch_size)
        self.alpha = alpha
        self.max_priority = 1.0
        self.tree_loc = 0
        self.tree_size = 2**math.ceil(math.log2(memory_size)) #ensures tree_size is a power of 2
        self.min_priorities_tree = MinSegmentTree(self.tree_size)
        self.priorities_tree = SumSegmentTree(self.tree_size)
    
    def store_transition(self, state, action, reward, next_state, done):
        '''Stores a transition step in memory'''
        super().store_transition(state, action, reward, next_state, done)
        importance = self.max_priority ** self.alpha
        self.priorities_tree[self.tree_loc] = importance
        self.min_priorities_tree[self.tree_loc] = importance
        self.tree_loc = (self.tree_loc+1) % self.memory_size
    
    def sample(self, beta=0.4):
        #Get indices for batch
        indices = []
        total_priority = self.priorities_tree.sum(0, len(self) - 1)
        segment = total_priority / self.batch_size
        for i in range(self.batch_size):
            a = segment*i
            b = segment*(i + 1)
            value = random.uniform(a, b)
            index = self.priorities_tree.find_prefixsum_idx(value)
            indices.append(index)
        
        #Weights
        weights = np.array([self.get_weight(index, beta) for index in indices])
        return dict(states=self.state_memory[indices], 
                    next_states=self.next_state_memory[indices],
                    actions=self.action_memory[indices],
                    rewards=self.reward_memory[indices],
                    dones=self.done_memory[indices],
                    weights=weights,
                    indices=indices)
        
    def update(self, indices, priorities):
        '''Updates the priorities of transitions[indices]'''
        assert len(indices) == len(priorities)
        for i, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= i < len(self)
            self.priorities_tree[i] = priority ** self.alpha
            self.min_priorities_tree[i] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
        
    def get_weight(self, index, beta):
        '''Gets the importance sampling weight for an index in segment trees'''
        min_priority = self.min_priorities_tree.min()/self.priorities_tree.sum()
        max_weight = (min_priority*len(self))**(-beta)
        priority = self.priorities_tree[index]/self.priorities_tree.sum()
        weight = (priority * len(self)) ** (-beta)
        weight = weight / max_weight
        return weight 
    
class NoisyLayer(nn.Module):
    ''' in_features = Number of input features
        out_features= Number of output features
        sigma_init  = Initial sigma (standard deviation) parameter'''
    def __init__(self,input_features, output_features, sigma=0.017):
        super(NoisyLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.sigma = sigma

        # Initialize parameters and buffers
        self.mean_weight = nn.Parameter(T.Tensor(output_features, input_features))
        self.std_dev_weight = nn.Parameter(T.Tensor(output_features, input_features))
        self.register_buffer('noise_weight', T.Tensor(output_features, input_features))

        self.mean_bias = nn.Parameter(T.Tensor(output_features))
        self.std_dev_bias = nn.Parameter(T.Tensor(output_features))
        self.register_buffer('noise_bias', T.Tensor(output_features))

        mu_range = 1 / math.sqrt(self.input_features)
        self.mean_weight.data.uniform_(-mu_range, mu_range)
        self.std_dev_weight.data.fill_(self.sigma / math.sqrt(self.input_features))
        self.mean_bias.data.uniform_(-mu_range, mu_range)
        self.std_dev_bias.data.fill_(self.sigma / math.sqrt(self.output_features))
        
        self.reset()
    
    def scale_noise(self, size):
        '''generates noise vectors by sampling from a factorized Gaussian distribution 
        with mean 0 and standard deviation 1, and then scales the noise vectors 
        according to the size of the input or output features.'''
        x = T.randn(size) # Noise tensor with random values from a standard normal distribution (mean=0, standard deviation=1).
        x = x.sign().mul(x.abs().sqrt()) #The sign(-1,1 or 0) of x multiplied by the square root of the absolute value of x
        return x
    
    def reset(self):
        epsilon_input, epsilon_output = map(self.scale_noise, [self.input_features, self.output_features])
        self.noise_weight.copy_(epsilon_output.ger(epsilon_input)) # ger gives the outer product of the epsilon out and the epsilon in
        self.noise_bias.copy_(epsilon_output)

    def forward(self, x: T.Tensor) -> T.Tensor:
        '''Takes parameters and creates a noisy layer of a nueral network, replace the forward use in
        a linear network, This is done in DuelingDeepQNetwork in self.fc1 onward'''
        return F.linear(
            x,
            self.mean_weight + self.std_dev_weight * self.noise_weight,
            self.mean_bias + self.std_dev_bias * self.noise_bias,
        )
    
    
        
class DuelingDeepQNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, nn_dims, atom_size, checkpoint_dir, name, support):
        """
        Initializes the DuelingDeepQNetwork agent.

        Args:
            input_dims (int): The number of input dimensions.
            output_dims (int): The number of output dimensions.
            nn_dims (int): The number of dimensions for the neural network layers.
            n_actions (int): The number of possible actions.
            atom_size (int): The number of atoms for the value distribution.
            checkpoint_dir (str): The directory to save checkpoints.
            name (str): The name of the checkpoint file.
            support (torch.Tensor): The support values for the value distribution.
        """
        super(DuelingDeepQNetwork, self).__init__()    # Inheriting from nn.Module

        self.support = support
        self.atoms = atom_size
        self.output_dims = output_dims

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)  # Save our checkpoint directory for later use

        self.feature = nn.Sequential(   #Feature layer that leads into advantage and value streams
            nn.Linear(input_dims, nn_dims),
            nn.ReLU(),
        )
        
        self.V_hidden = NoisyLayer(nn_dims, nn_dims) # Value Stream
        self.V = NoisyLayer(nn_dims, atom_size)
        
        self.A_hidden = NoisyLayer(nn_dims, nn_dims) # Advantage Stream
        self.A = NoisyLayer(nn_dims, atom_size * output_dims)

    def forward(self, x: T.tensor):
        dist = self.distrib(x)
        q = T.sum(dist * self.support, dim=2)
        return q
    
    def distrib(self,x):
        feature = self.feature(x)
        A = F.relu(self.A_hidden(feature))
        V = F.relu(self.V_hidden(feature))
        A = self.A(A).view(-1, self.output_dims, self.atoms)
        V = self.V(V).view(-1, 1, self.atoms)
        q_atoms = V + A - A.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)
        return dist
    
    def reset(self):
        self.A_hidden.reset()
        self.V_hidden.reset()
        self.A.reset()
        self.V.reset()
    
    #Save and load checkpoints
    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, file):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
    


class DQNAgent():
    def __init__(self, env: gymnasium.Env, learning_rate, batch_size, gamma, epsilon, alpha=0.6, beta=0.4,per_const=1e-6, max_memory_size=100000, hidden_neurons=64, eps_min=0.1, replace=1000, checkpoint_dir='tmp/', min_return_value = 0.0, max_return_value = 200.0, atom_size = 51): 
        # Adjust epsilon decay rate later, right now linear decay

        self.env = env
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.observation_space = env.observation_space
        self.observation_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.action_space = [i for i in range(self.n_actions)]
        self.memory_size = max_memory_size
        
        self.per_const = per_const
        self.beta = beta
        self.memory = PrioritisedReplay(self.observation_shape[0], max_memory_size, batch_size, alpha)
        
        self.epsilon = epsilon
        self.eps_max = epsilon
        self.eps_min = eps_min
        self.gamma = gamma

        self.replace_target_count = replace
        self.checkpoint_dir = checkpoint_dir

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        # Categorical DQN params
        self.min_return_value = min_return_value
        self.max_return_value = max_return_value
        self.atoms = atom_size
        self.value_distribution = T.linspace(self.min_return_value, self.max_return_value, self.atoms).to(self.device)

        # Q Evaluation Network
        self.network = DuelingDeepQNetwork(input_dims=self.observation_shape[0], output_dims=self.n_actions, nn_dims=hidden_neurons,
                                    atom_size=atom_size, name='surround_dueling_ddqn',
                                    checkpoint_dir=self.checkpoint_dir, support=self.value_distribution)
        
        self.target_network = DuelingDeepQNetwork(input_dims=self.observation_shape[0], output_dims=self.n_actions, nn_dims=hidden_neurons,
                                    atom_size=atom_size, name='surround_dueling_ddqn_target',
                                    checkpoint_dir=self.checkpoint_dir, support=self.value_distribution)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        
        self.optimiser = optim.Adam(self.network.parameters())
        
        self.transition = []
        
        self.testing = False
         
    def action(self, state: np.ndarray):
        '''Action choice based on epsilon-greedy policy, returns the action as a np.ndarray''' 
        action = self.network(T.FloatTensor(state).to(self.device)).argmax()
        action = action.detach().cpu().numpy() 
        
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
        pre_loss = self.calculate_loss(batches)
        loss = T.mean(pre_loss*T.FloatTensor(batches["weights"].reshape(-1, 1)).to(self.device))
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1) # clips gradients between -1 and 1, can change
        self.optimiser.step()
        
        priority_loss = pre_loss.detach().cpu().numpy()
        new_priorities = priority_loss + self.per_const
        self.memory.update(batches['indices'], new_priorities)
        self.network.reset()
        self.target_network.reset()
        return loss.item()

    def calculate_loss(self, samples):
        """loss function to simplify learn() function"""
        state = T.FloatTensor(samples["states"]).to(self.device)
        next_state = T.FloatTensor(samples["next_states"]).to(self.device)
        action = T.LongTensor(samples["actions"]).to(self.device)
        reward = T.FloatTensor(samples["rewards"].reshape(-1, 1)).to(self.device)
        done = T.FloatTensor(samples["dones"].reshape(-1, 1)).to(self.device)
        delta = float(self.max_return_value - self.min_return_value) / (self.atoms - 1)
        
        with T.no_grad():
            next_action = self.network(next_state).argmax(1)
            next_dist = self.target_network.distrib(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            target_return = reward + (1 - done) * self.gamma * self.value_distribution
            target_return = target_return.clamp(min=self.min_return_value, max=self.max_return_value)
            support_indices = (target_return - self.min_return_value) / delta
            lower_bound = support_indices.floor().long()
            upper_bound = support_indices.ceil().long()

            offset = (T.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.atoms).to(self.device))

            proj_dist = T.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (lower_bound + offset).view(-1), (next_dist * (upper_bound.float() - support_indices)).view(-1))
            proj_dist.view(-1).index_add_(0, (upper_bound + offset).view(-1), (next_dist * (support_indices - lower_bound.float())).view(-1))

        dist = self.network.distrib(state)
        lol = dist[range(self.batch_size), action]
        log_p = T.log(lol)
        loss = -(proj_dist * log_p).sum(1)

        return loss

    def train(self, steps):
        self.testing = False
        tracked_info = {
            "scores":[],
            "losses":[],
            "epsilons":[]
        }
        learn_count = 0
        episodes = 0
        score = 0
        state, _ = self.env.reset()
        for step in tqdm(range(1,steps+1)):
            action = self.action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            state = next_state
            done = terminated or truncated
            score += reward
            
            self.beta = self.beta + min(step/steps,1)*(1-self.beta)
            
            if not self.testing:
                self.transition += [reward, next_state, done]
                self.memory.store_transition(*self.transition)
            
            if done:
                state, _ = self.env.reset()
                tracked_info["scores"].append(score)
                score = 0
                episodes+=1
                if episodes > 10 and episodes % 10 == 0:
                    self.save_models()
            
            if len(self.memory) >= self.batch_size:
                loss = self.learn()
                tracked_info["losses"].append(loss)
                learn_count += 1
                tracked_info["epsilons"].append(self.epsilon)
                self.replace_target_network(learn_count)
                
        self.env.close()
        return tracked_info
                
            

