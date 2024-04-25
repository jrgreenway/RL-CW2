#Imports
#from DQN import DQN #Change According to NN naming
import torch as T
import torch.nn as nn               # Neural Network    
# We may need to import something for convulutional layers because we are working with images
import torch.nn.functional as F     # Activation Functions
import torch.optim as optim         # Optimiser
import numpy as np                  # Array Manipulation

# This inmplementation has no target network or convulutional layers
# Convulutional layers are used to process the image and extract features from it
# Target network is basically a copy of the main network that is updated every few steps

class DeepQNetwork(nn.Module):
    # lr            = learning rate
    # input_dims    = input dimensions
    # fc1_dims      = fully connected layer 1 dimensions
    # fc2_dims      = fully connected layer 2 dimensions
    # n_actions     = number of actions
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()    # Calls nn.Module __init__ function to set up PyTorch stuff
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)   # *self.input_dims is the same as self.input_dims[0], self.input_dims[1], etc.
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)      # Create a fully connected layer with fc1_dims inputs and fc2_dims outputs
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)  # Create an Adam optimizer to adjust weights of the nn with the learning rate lr
        self.loss = nn.MSELoss()  # Mean Squared Error Loss function. 2 inputs: input and target. Returns the mean squared error between the input and target
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # Use GPU if your PC has one (better), if not use CPU
        self.to(self.device)    # Move the nn to the device (GPU or CPU)

    # We don't need to handle backpropagation manually because PyTorch does it for us
    # Forward function is called when you pass data through the nn
    def forward(self, state):
        x = F.relu(self.fc1(state))     # Pass state into first layer and apply ReLU activation function in each neuron in the layer
        x = F.relu(self.fc2(x))         # Pass output from 1st layer into 2nd layer
        actions = self.fc3(x)           # Pass output from 2nd layer into 3rd layer but don't apply action function, just save it
        return actions
    



class DQNAgent():
    # observation_Space = Number of inputs  (same as input_dims)
    # action_space = Number of outputs (like n_actions)
    # hidden_neurons = Number of neurons in the hidden layers (same as fc1_dims and fc2_dims)
    def __init__(self, learning_rate=0, memory_capacity=0, batch_size=0, observation_space=0, 
                  n_actions=0, hidden_neurons=128, max_memory_size=100000, epsilon=0.3, eps_decay=5e-4, eps_min=0.01, gamma=0.95) -> None: 
        # Adjust epsilon decay rate later, right now linear decay

        self.learning_rate = learning_rate
        self.memory_capacity = memory_capacity
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
        self.Q_eval = DeepQNetwork(self.learning_rate, input_dims=self.observation_space, fc1_dims=self.hidden_neurons,
                                    fc2_dims=self.hidden_neurons, n_actions=self.n_actions)


        # Experience memory
        # The video guide I(Axel) followed uses multiple arrays instead of a deque. 1 for each experience: (state, action, reward, new_state, terminal?)
        self.state_memory = np.zeroes((self.memory_size, *observation_space), dtype=np.float32)
        self.action_memory = np.zeroes(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeroes(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeroes(self.memory_size, dtype=np.bool)
        self.new_state_memory = np.zeroes((self.memory_size, *observation_space), dtype=np.float32)

        
        #Policy Network
        #self.network = 
        #self.target_network = 
        #self.target_network.load_state_dict(self.network.state_dict())
        #self.target_network.eval()
        #self.network.eval()
        
        #Network Optimiser - Have a look at different optimisers
        #The optimiser updates the weightings in the nn to predict better q-values.
        #self.optimiser = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
    
    def store_transition(self, state, action, reward, next_state, done):
        #Store transitions in memory so that we can use them for experience replay
        index = self.memory_counter % self.memory_size    # Wraps around when memory is full, overwriting oldest memories
        self.state_memory[index]      = state
        self.action_memory[index]     = action
        self.reward_memory[index]     = reward
        self.new_state_memory[index]  = next_state
        self.terminal_memory[index]   = done
        self.memory_counter          += 1
        
        
    def action(self, state): # Chooses action
        if np.random.random(1) > self.epsilon: # Best action (Exploitation)
            state = T.tensor(state, dtype=T.float).to(self.Q_eval.device) # Convert state to tensor,prepare Q_eval for preprocessing
            actions = self.Q_eval.forward(state) # actions is all the Q-value outputs from the network after a forward pass
            action = T.argmax(actions).item() # Get the action with the highest Q-value in actions
        else: # Random action (Exploration)
            action = np.random.choice(self.action_space) # Choose a random action from the action space

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
    