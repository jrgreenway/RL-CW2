#Imports
#from DQN import DQN #Change According to NN naming
import torch as T
import torch.nn as nn               # Neural Network (nn)
# We may need to import something for convulutional layers because we are working with images
import torch.nn.functional as F     # Activation Functions
import torch.optim as optim         # nn Optimiser
import numpy as np                  # numpy
from termcolor import colored       # Colored text for debugging


# This inmplementation has no target network or convulutional layers
# Convulutional layers are used to process the image and extract features from it
# Target network is basically a copy of the main network that is updated every few steps

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.n_actions = n_actions

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(self.input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # Calculate the output dimensions after the convolutional layers
        self.conv_output_dims = self.calculate_conv_output_dims()

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.conv_output_dims, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self):
        # Create a dummy tensor with the input dimensions
        dummy = T.zeros(1,*self.input_dims)

        # Pass the dummy tensor through the convolutional layers
        dummy = self.conv1(dummy)
        dummy = self.conv2(dummy)
        dummy = self.conv3(dummy)

        # Return the product of the dimensions of the output tensor
        return int(np.prod(dummy.size()))

    def forward(self, state):
        # Reshape the state to match the expected input dimensions of the convolutional layers
        #state = state.view(-1, *self.input_dims)

        # Pass the state through the convolutional layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from the convolutional layers
        x = x.view(x.size()[0], -1)

        # Pass the output through the fully connected layers
        x = F.relu(self.fc1(x))
        actions = self.fc2(x)

        return actions
    



class DQNAgent():
    # observation_Space = Number of inputs  (same as input_dims)
    # action_space = Number of outputs (like n_actions)
    # hidden_neurons = Number of neurons in the hidden layers (same as fc1_dims and fc2_dims)
    def __init__(self, learning_rate, batch_size, observation_space, 
                  n_actions, gamma, epsilon, max_memory_size=10000, hidden_neurons=64, eps_decay=0.99, eps_min=0.01, frames=4) -> None: 
        # Adjust epsilon decay rate later, right now linear decay

        self.frames = frames
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

        self.memory_counter = 0 # Counter to keep track of how many memories we have stored

        # Q Evaluation Network
        # Hidden neurons defaulted to 128 (for testing)
        self.Q_eval = DeepQNetwork(self.learning_rate, input_dims=self.observation_space[1:], n_actions=self.n_actions)


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
            state = T.tensor(observation, dtype=T.float).to(self.Q_eval.device) # Convert state to tensor,prepare Q_eval for preprocessing
            actions = self.Q_eval.forward(state) # actions is all the Q-value outputs from the network after a forward pass
            action = T.argmax(actions).item() # Get the action with the highest Q-value in actions
        else: # Random action (Exploration)
            action = np.random.choice(self.action_space) # Choose a random action from the action space

        return action
    

    def learn(self):
        # Learn from batch of experience . Experience Replay!
        # Start learning as soon as we've filled a batch of memories (to sample from when learning)
        if self.memory_counter < self.batch_size: # 
            return
        
        self.Q_eval.optimizer.zero_grad()      # Set up optimizer (Zero the gradients of it)

        max_mem = min(self.memory_counter, self.memory_size)                # Max memory we can use
        batch = np.random.choice(max_mem, self.batch_size, replace=False)       # Randomly take *batch_size* numbers from range(max_mem)
        # Batch is an array of indices we will use to sample from memory
                      # Create an array of numbers from 0 to batch_size

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device).reshape(-1, *self.state_memory[batch].shape[2:])         # Convert all the bacthes to tensors
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device).reshape(-1, *self.new_state_memory[batch].shape[2:])  
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device).repeat_interleave(self.frames)        
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device).repeat_interleave(self.frames)   

        action_batch = T.tensor(self.action_memory[batch]).to(self.Q_eval.device)
        action_batch_repeated = action_batch.repeat_interleave(self.frames)
        batch_index_grouped = T.arange(self.batch_size*self.frames).to(self.Q_eval.device)
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index_grouped, action_batch_repeated] # Shape [64]  # Get the Q-values of the actions we took
        q_next = self.Q_eval.forward(new_state_batch) # SHape: [64, 3]                   # Pass a bactch of states into the network to get Q-values for each action in each state
        q_next[terminal_batch] = 0.0                                                     #  Set Q-values of all terminal states to 0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]                     # Calculate the target Q-values using the Bellman equation
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)                    # Mean-squared error between q_next and q_eval
        loss.backward()       # Backpropagate the loss. Tells us how much the weights of the nn affect the loss. Helps us adjust the weights to minimize loss
        self.Q_eval.optimizer.step()   # Step the optimizer. Adjust the weights of the nn based on the gradients calculated in the loss.backward() step

        # Decrement epsilon (explore rate)
        

    def decay(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

        
    # Axel: I havent added a target nn yet
    # def update_target(self):
    #     self.target_network.load_state_dict(self.network.state_dict())
        

