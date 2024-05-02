#Imports
#from DQN import DQN #Change According to NN naming
import math
import torch as T
import torch.nn as nn               # Neural Network (nn)
# We may need to import something for convulutional layers because we are working with images
import torch.nn.functional as F     # Activation Functions
import torch.optim as optim         # nn Optimiser
import numpy as np                  # numpy
from termcolor import colored       # Colored text for debugging
import os                           # For file joining operations to handle model checkpointing

# This inmplementation has no target network or convulutional layers
# Convulutional layers are used to process the image and extract features from it
# Target network is basically a copy of the main network that is updated every few steps

# 
class NoisyLinear(nn.Module):
    # in_features = Number of input features
    # out_features= Number of output features
    # sigma_init  = Initial sigma (standard deviation) parameter
    def __init__(self,in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Make these tensors as opposed to empties?
        self.weight_mu = nn.Parameter(T.empty(out_features, in_features)) # Mean value weight parameter
        self.weight_sigma = nn.Parameter(T.empty(out_features, in_features)) # Standard Deviation value weight parameter
        self.register_buffer('weight_epsilon', T.empty(out_features, in_features)) # Buffer that holds noise values added to weights during training

        self.bias_mu = nn.Parameter(T.empty(out_features)) # Mean value bias parameter
        self.bias_sigma = nn.Parameter(T.empty(out_features)) # Standard Deviation value bias parameter
        self.register_buffer('bias_epsilon', T.empty(out_features)) # Buffer that holds noise values added to biases during training

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Reset the learnable network parameters (mu and sigma), 
        # Uses factorized gaussian noise, more computationally efficient
        # The below are just ranges that are reasonable and a value is chosen from them to initialise mu and sigma
        # These ranges could be changed but are probably the most reasonable
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range) # Chooses with equal chance a value in the range
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        # 
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # ger gives the outer product of the epsilon out and the epsilon in
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in)) 
        self.bias_epsilon.copy_(epsilon_out)

    # Missing forward right now....

    # The scale_noise() method generates noise vectors by sampling from a 
    # factorized Gaussian distribution with mean 0 and standard deviation 1, 
    # and then scales the noise vectors according to the size of the input or output features.
    # 
    def scale_noise(self, size):
        x = T.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    

class DuelingDeepQNetwork(nn.Module):
    # lr            = learning rate
    # input_dims    = input dimensions
    # fc1_dims      = fully connected layer 1 dimensions
    # fc2_dims      = fully connected layer 2 dimensions
    # n_actions     = number of actions
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, checkpoint_dir, name):
        super(DuelingDeepQNetwork, self).__init__()    # Calls nn.Module __init__ function to set up PyTorch stuff
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)  # Save our checkpoint directory for later use

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)   # *self.input_dims is the same as self.input_dims[0], self.input_dims[1], etc.
        self.V = nn.Linear(fc1_dims, 1)             # Value stream: tells the agent theh value of thhe current state
        self.A = nn.Linear(fc1_dims, n_actions)     # Advantage stream: tells the agent the relative advantage of eachh action in a given state
        #self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)      # Create a fully connected layer with fc1_dims inputs and fc2_dims outputs
        #self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)  # Create an Adam optimizer to adjust weights of the nn with the learning rate lr
        self.loss = nn.MSELoss()  # Mean Squared Error Loss function. 2 inputs: input and target. Returns the mean squared error between the input and target
        #print(f"GPU available?" + colored(T.cuda.is_available(), 'green'))
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # Use GPU if your PC has one (better), if not use
        self.device = T.device('cpu')  # Use CPU
        self.to(self.device)    # Move the nn to the device (GPU or CPU)

    # We don't need to handle backpropagation manually because PyTorch does it for us
    # Forward function is called when you pass data through the nn
    def forward(self, state):
        x = F.relu(self.fc1(state))     # Pass state into first layer and apply ReLU activation function in each neuron in the layer
        # x = F.relu(self.fc2(x))         # Pass output from 1st layer into 2nd layer
        # actions = self.fc3(x)           # Pass output from 2nd layer into 3rd layer but don't apply action function, just save it
        V = self.V(x)
        A = self.A(x)
        
        return V, A
    
    #Save and load checkpoints
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, file):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class DQNAgent():
    # observation_Space = Number of inputs  (same as input_dims)
    # action_space = Number of outputs (like n_actions)
    # hidden_neurons = Number of neurons in the hidden layers (same as fc1_dims and fc2_dims)
    def __init__(self, learning_rate, batch_size, observation_space, 
                  n_actions, gamma, epsilon, max_memory_size=100000, hidden_neurons=64, eps_decay=5e-4, eps_min=0.01, replace=1000, checkpoint_dir='tmp/dueling_ddqn'): 
        # Adjust epsilon decay rate later, right now linear decay

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.observation_space = observation_space
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)] # Represents all possible actions. Makes it easier to select random actions
        self.hidden_neurons = hidden_neurons
        self.memory_size = max_memory_size
        
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma

        self.replace_target_count = replace
        self.checkpoint_dir = checkpoint_dir

        self.learn_step_counter = 0

        self.memory_counter = 0 # Counter to keep track of how many memories we have stored

        # Q Evaluation Network
        # Hidden neurons defaulted to 128 (for testing)
        self.Q_eval = DuelingDeepQNetwork(self.learning_rate, input_dims=self.observation_space, fc1_dims=self.hidden_neurons,
                                    fc2_dims=self.hidden_neurons, n_actions=self.n_actions, name='surround_dueling_ddqn_q_eval',
                                    checkpoint_dir=self.checkpoint_dir)
        
        self.Q_next = DuelingDeepQNetwork(self.learning_rate, input_dims=self.observation_space, fc1_dims=self.hidden_neurons,
                                    fc2_dims=self.hidden_neurons, n_actions=self.n_actions, name='surround_dueling_ddqn_q_eval',
                                    checkpoint_dir=self.checkpoint_dir)

        # Experience memory
        # The video guide I(Axel) followed uses multiple arrays instead of a deque. 1 for each experience: (state, action, reward, new_state, terminal?)
        self.state_memory = np.zeros((self.memory_size, *observation_space), dtype=np.float32) ### DEBUGGING: Obersvation_space = [2]
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)
        self.truncated_memory = np.zeros(self.memory_size, dtype=bool)
        self.new_state_memory = np.zeros((self.memory_size, *observation_space), dtype=np.float32)

        
        #Policy Network
        #self.network = 
        #self.target_network = 
        #self.target_network.load_state_dict(self.network.state_dict())
        #self.target_network.eval()
        #self.network.eval()
        
        #Network Optimiser - Have a look at different optimisers
        #The optimiser updates the weightings in the nn to predict better q-values.
        #self.optimiser = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
    
    def store_transition(self, state, action, reward, next_state, terminated, truncated):
        #Store transitions in memory so that we can use them for experience replay
        index = self.memory_counter % self.memory_size    # Wraps around when memory is full, overwriting oldest memories
        self.state_memory[index]      = state
        self.action_memory[index]     = action
        self.reward_memory[index]     = reward
        self.new_state_memory[index]  = next_state
        self.terminal_memory[index]   = terminated
        self.truncated_memory[index]  = truncated

        self.memory_counter          += 1
         
    def action(self, observation): # Chooses action
        if np.random.random(1) > self.epsilon: # Best action (Exploitation)
            state = T.tensor([observation], dtype=T.float).to(self.Q_eval.device) # Convert state to tensor,prepare Q_eval for preprocessing
            _, advantage = self.Q_eval.forward(state) # Push state through network and get advantage function
            # actions = self.Q_eval.forward(state) # actions is all the Q-value outputs from the network after a forward pass
            action = T.argmax(advantage).item() # Get the action with the highest advantage
        else: # Random action (Exploration)
            action = np.random.choice(self.action_space) # Choose a random action from the action space

        return action
    
    def replace_target_network(self):   # Handles replacing the target network
        if self.learn_step_counter % self.replace_target_count == 0:  # Do so after certain number of steps
            self.Q_next.load_state_dict(self.Q_eval.state_dict())     # Load the state dictionary from the evaluation q network to the q_next network

    def save_models(self):
        self.Q_eval.save_checkpoint()
        self.Q_next.save_checkpoint()

    def load_models(self):
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()

    def learn(self):
        # Learn from batch of experience . Experience Replay!
        # Start learning as soon as we've filled a batch of memories (to sample from when learning)
        if self.memory_counter < self.batch_size: # 
            return
        
        self.Q_eval.optimizer.zero_grad()      # Set up optimizer (Zero the gradients of it)

        self.replace_target_network() # Want to replace our target network 

        max_mem = min(self.memory_counter, self.memory_size)                # Max memory we can use
        batch = np.random.choice(max_mem, self.batch_size, replace=False)       # Randomly take *batch_size* numbers from range(max_mem)
        # Batch is an array of indices we will use to sample from memory
        batch_index = np.arange(self.batch_size, dtype=np.int32)               # Create an array of numbers from 0 to batch_size

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)          # Convert all the bacthes to tensors
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)  
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)        
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)   

        action_batch = self.action_memory[batch]                                         # Get the actions from the batch

        V_s, A_s = self.Q_eval.forward(state_batch)
        V_s_, A_s_ = self.Q_next.forward(new_state_batch)

        V_s_eval, A_s_eval = self.Q_eval.forward(new_state_batch)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[batch_index, action_batch]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma*q_next[batch_index, max_actions]


        #q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch] # Shape [64]  # Get the Q-values of the actions we took
        #q_next = self.Q_eval.forward(new_state_batch) # SHape: [64, 3]                   # Pass a bactch of states into the network to get Q-values for each action in each state
        #q_next[terminal_batch] = 0.0                                                     #  Set Q-values of all terminal states to 0

        #q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]                     # Calculate the target Q-values using the Bellman equation
        
        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)                    # Mean-squared error between q_next and q_eval
        loss.backward()       # Backpropagate the loss. Tells us how much the weights of the nn affect the loss. Helps us adjust the weights to minimize loss
        self.Q_eval.optimizer.step()   # Step the optimizer. Adjust the weights of the nn based on the gradients calculated in the loss.backward() step
        self.learn_step_counter += 1

        # Decrement epsilon (explore rate)
        self.epsilon = self.epsilon - self.eps_decay if self.epsilon > self.eps_min else self.eps_min

        
        
    # Axel: I havent added a target nn yet
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
        

