#Imports
from collections import deque
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
from collections import deque
import logging
from tqdm import tqdm
from termcolor import colored


class nStepReplayMemory:
    """Converts to n-step transitions and stores them in a cyclic buffer."""
    # observation_dimensions: int, memory_size: int, batch_size: int = 32, n_step: int = 3, gamma: float = 0.99)
    def __init__(self, observation_dimensions, memory_size, batch_size=32, n_step=3, gamma=0.99):

        self.state_memory = np.zeros([memory_size, observation_dimensions], dtype=np.float32)
        self.next_state_memory = np.zeros([memory_size, observation_dimensions], dtype=np.float32)
        self.action_memory = np.zeros([memory_size], dtype=np.int32)
        self.reward_memory = np.zeros([memory_size], dtype=np.float32)
        self.done_memory = np.zeros(memory_size, dtype=bool)
        self.max_memory_size = memory_size    # max capacity of the buffer
        self.batch_size = batch_size
        self.memory_index = 0    # pointer to the current location in the buffer
        self.size = 0   # current size of the buffer
        
        # for N-step
        self.n_step_buffer = deque(maxlen=n_step)   
        self.n_step = n_step
        self.gamma = gamma

    # STORE WITHOUT END STEP (FOR TESTING)
    def OLDReplyMemoryStoreTransition(self, state, action, reward, next_state, done):
        #This is how the old ReplayMemory class stored transitions
        self.state_memory[self.memory_index] = state
        self.action_memory[self.memory_index] = action
        self.reward_memory[self.memory_index] = reward
        self.next_state_memory[self.memory_index] = next_state
        self.done_memory[self.memory_index] = done
        self.memory_index = (self.memory_index + 1) % self.max_memory_size
        self.size = min(self.size + 1, self.max_memory_size)

    # store the transition in the buffer and return the n-step transition if ready
    # input experience - output n-step transition
    # state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    def nStep_store(self, state, action, reward, next_state, done):
        
        experience = (state, action, reward, next_state, done)  # single step transition
        self.n_step_buffer.append(experience)

        if len(self.n_step_buffer) < self.n_step:   # if n_step_buffer is not full do not return anything
            return ()
        
        # Calculate n-step return
        last_experience = self.n_step_buffer[-1]           # latest transition in buffer
        _, _, reward_nstep, next_state_nstep, done_nstep = last_experience

        for experience in reversed(list(self.n_step_buffer)[:-1]): # Iterate backwards, excluding the last experience
            r, n_s, d = experience[-3:]
            reward_nstep = r + self.gamma * reward_nstep * (1 - d)
            next_state_nstep, done_nstep = (n_s, d) if d else (next_state_nstep, done_nstep) # if any of the transitions is done, the n-step return is done    
        
        reward, next_state, done = reward_nstep, n_s, d      # n-step return  
        state, action = self.n_step_buffer[0][:2]  # state and action become the first state and action in the n-step buffer
        
        # store the n-step transition in the memory
        self.state_memory[self.memory_index] = state
        self.next_state_memory[self.memory_index] = next_state
        self.action_memory[self.memory_index] = action
        self.reward_memory[self.memory_index] = reward
        self.done_memory[self.memory_index] = done
        self.memory_index = (self.memory_index + 1) % self.max_memory_size
        self.size = min(self.size + 1, self.max_memory_size)
        
        return self.n_step_buffer[0] # return the single step transition

    def sample_batch(self):
        #Returns a batch of random experiences of size batch_size
        batch_indexes = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(states=self.state_memory[batch_indexes], 
                    next_states=self.next_state_memory[batch_indexes],
                    actions=self.action_memory[batch_indexes],
                    rewards=self.reward_memory[batch_indexes],
                    dones=self.done_memory[batch_indexes])
    
    def __len__(self):
        return self.size
    
class NoisyLayer(nn.Module):
    ''' in_features = Number of input features
        out_features= Number of output features
        sigma  = Initial sigma (standard deviation) parameter'''
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
    def __init__(
            self,
            env: gymnasium.Env,
            learning_rate,
            batch_size,
            gamma,
            min_return_value,
            max_return_value,
            alpha=0.6,
            beta=0.4,
            per_const=1e-6,
            max_memory_size=100000,
            hidden_neurons=128,
            replace_target_nn=1000,
            checkpoint_dir='tmp/',
            n_step = 3, # n-step learning
            atom_size = 51, 
            log=True
        ): 
        # Adjust epsilon decay rate later, right now linear decay

        self.env = env
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.observation_space = env.observation_space
        self.observation_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.action_space = [i for i in range(self.n_actions)]
        self.memory_size = max_memory_size
        self.n_step = n_step
        
        self.per_const = per_const
        self.beta = beta
        self.memory = nStepReplayMemory(self.observation_shape[0], max_memory_size, batch_size, n_step=n_step, gamma=gamma)
        
        self.gamma = gamma

        self.replace_target_count = replace_target_nn
        self.checkpoint_dir = checkpoint_dir

        # 1 step and n step learning

        # memory for 1-step Learning
        #self.memory = nStepReplayBuffer(self.observation_shape[0], max_memory_size, batch_size, n_step=1, gamma=gamma)
        
        # memory for N-step Learning
        """
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = nStepReplayBuffer(self.observation_shape[0], max_memory_size, batch_size, n_step=n_step, gamma=gamma)
        """

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        print(colored(f"Using {self.device}", "green"))
        
        # Categorical DQN params
        self.min_return_value = min_return_value
        self.max_return_value = max_return_value
        self.atoms = atom_size
        self.value_distribution = T.linspace(self.min_return_value, self.max_return_value, self.atoms).to(self.device)
        

        # Q Evaluation Network
        self.network = DuelingDeepQNetwork(input_dims=self.observation_shape[0], output_dims=self.n_actions, nn_dims=hidden_neurons,
                                    atom_size=atom_size, name='surround_dueling_ddqn',
                                    checkpoint_dir=self.checkpoint_dir, support=self.value_distribution).to(self.device)
        
        self.target_network = DuelingDeepQNetwork(input_dims=self.observation_shape[0], output_dims=self.n_actions, nn_dims=hidden_neurons,
                                    atom_size=atom_size, name='surround_dueling_ddqn_target',
                                    checkpoint_dir=self.checkpoint_dir, support=self.value_distribution).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        
        # Optimser
        self.optimiser = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        self.transition = []
        
        self.testing = False
        
        self.log = log
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
        batches = self.memory.sample_batch()
        loss = self.calculate_loss(batches)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1) # clips gradients between -1 and 1, can change
        self.optimiser.step()
        
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
        loss = -(proj_dist * log_p).sum(1).mean()

        return loss

    def make_params_dict(self):
        return dict(
            env=self.env.spec.id,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gamma=self.gamma,
            max_memory_size=self.memory_size,
            replace=self.replace_target_count,
            min_return_value=self.min_return_value,
            max_return_value=self.max_return_value,
            per_const=self.per_const,
            atom_size=self.network.atoms,
            n_step=self.n_step
        )

    def train(self, steps):
        self.testing = False
        tracked_info = {
            "scores":[],
            "losses":[]
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
            
            self.beta = self.beta + min(step/steps,1)*(1-self.beta)
            
            if not self.testing:
                self.transition += [reward, next_state, done]
                self.memory.nStep_store(*self.transition)    ## Prioritised replay
            
            if done:
                state, _ = self.env.reset()
                tracked_info["scores"].append(score)
                if self.log:
                    self.logger.info(f"Ep. Num.: {episodes}, Ep. Score: {score}, Avg. Score: {np.mean(tracked_info['scores'][-10:])}")
                score = 0
                episodes+=1
                if episodes > 10 and episodes % 50 == 0:
                    self.save_models()
            
            if len(self.memory) >= self.batch_size:
                loss = self.learn()
                tracked_info["losses"].append(loss)
                learn_count += 1
                self.replace_target_network(learn_count)
                
        self.env.close()
        return tracked_info, parameters
    
        
                
            