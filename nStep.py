import os
from collections import deque
from typing import Deque, Dict, List, Tuple
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from IPython.display import clear_output



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
        # STORE NSTATE EXPERIENCE INSTEAD  
        self.state_memory[self.memory_index] = state
        self.action_memory[self.memory_index] = action
        self.reward_memory[self.memory_index] = reward
        self.next_state_memory[self.memory_index] = next_state
        self.done_memory[self.memory_index] = done
        self.memory_index = (self.memory_index + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

    def __len__(self):
        return self.size
    




class nStepReplayMemory:
    """Converts to n-step transitions and stores them in a cyclic buffer."""

    #
    def __init__(self, observation_dimensions: int, memory_size: int, batch_size: int = 32, n_step: int = 3, gamma: float = 0.99):

        self.state_memory = np.zeros([memory_size, observation_dimensions], dtype=np.float32)
        self.next_state_memory = np.zeros([memory_size, observation_dimensions], dtype=np.float32)
        self.action_memory = np.zeros([memory_size], dtype=np.int32)
        self.reward_memory = np.zeros([memory_size], dtype=np.float32)
        self.done_memory = np.zeros(memory_size, dtype=bool)
        self.max_memory_size = memory_size    # max capacity of the buffer
        self.batch_size = batch_size
        self.memory_index = 0    # pointer to the current location in the buffer
        self.size = 0   # current size of the buffer
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)   
        self.n_step = n_step
        self.gamma = gamma

    def ReplyMemoryStoreTransition(self, state, action, reward, next_state, done):
        #This is how the old ReplayMemory class stored transitions
        self.state_memory[self.memory_index] = state
        self.action_memory[self.memory_index] = action
        self.reward_memory[self.memory_index] = reward
        self.next_state_memory[self.memory_index] = next_state
        self.done_memory[self.memory_index] = done
        self.memory_index = (self.memory_index + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)


    # store the transition in the buffer and return the n-step transition if ready
    # input experience - output n-step transition
    # state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    def store(self, state, action, reward, next_state, done):
        
        experience = (state, action, reward, next_state, done)  # single step transition
        self.n_step_buffer.append(experience)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:   # if n_step_buffer is not full do not return anything
            return ()
        
        # make a n-step transition.     Look into the future
        reward, next_state, done = self._get_n_step_info(self.n_step_buffer, self.gamma) # Look into the future, sets n-step values
        state, action = self.n_step_buffer[0][:2]  # state and action become the first state and action in the n-step buffer
        
        self.state_memory[self.memory_index] = state
        self.next_state_memory[self.memory_index] = next_state
        self.action_memory[self.memory_index] = action
        self.reward_memory[self.memory_index] = reward
        self.done_memory[self.memory_index] = done
        self.memory_index = (self.memory_index + 1) % self.max_memory_size
        self.size = min(self.size + 1, self.max_memory_size)
        
        return self.n_step_buffer[0] # return the single step transition

    def sample_batch(self) -> Dict[str, np.ndarray]:
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.state_memory[indices],
            next_state=self.next_state_memory[indices],
            acts=self.action_memory[indices],
            rews=self.reward_memory[indices],
            done=self.done_memory[indices],
            # for N-step Learning
            indices=indices,
        )
    
    def sample_batch_from_idxs(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.state_memory[indices],
            next_state=self.next_state_memory[indices],
            acts=self.action_memory[indices],
            rews=self.reward_memory[indices],
            done=self.done_memory[indices],
        )
    
    def _get_n_step_info(self, n_step_buffer: Deque, gamma: float) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step reward, next_state, and done."""
        # info of the last transition
        r, next_state, done = n_step_buffer[-1][-3:]  # n_step_buffer[-1][-3:] means 

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            r = r + gamma * r * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return r, next_state, done

    def __len__(self) -> int:
        return self.size