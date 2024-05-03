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


# THIS is the n step classes
class ReplayBuffer:
    """simple numpy replay buffer."""

    def __init__(self, observation_dimensions: int, size: int, batch_size: int = 32, n_step: int = 3, gamma: float = 0.99):
        self.obs_buf = np.zeros([size, observation_dimensions], dtype=np.float32)
        self.next_state_buf = np.zeros([size, observation_dimensions], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size = size    # max capacity of the buffer
        self.batch_size = batch_size
        self.ptr = 0    # pointer to the current location in the buffer
        self.size = 0   # current size of the buffer
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)   
        self.n_step = n_step
        self.gamma = gamma

    # store the transition in the buffer and return the n-step transition if ready
    # input experience - output n-step transition
    def store(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        
        experience = (state, action, reward, next_state, done)  # single step transition
        self.n_step_buffer.append(experience)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:   # if n_step_buffer is not full do not return anything
            return ()
        
        # make a n-step transition.     Look into the future
        reward, next_state, done = self._get_n_step_info(self.n_step_buffer, self.gamma) # Look into the future, sets n-step values
        state, action = self.n_step_buffer[0][:2]  # state and action become the first state and action in the n-step buffer
        
        self.obs_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.acts_buf[self.ptr] = action
        self.rews_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0] # return the single step transition

    def sample_batch(self) -> Dict[str, np.ndarray]:
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[indices],
            next_state=self.next_state_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            # for N-step Learning
            indices=indices,
        )
    
    def sample_batch_from_idxs(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[indices],
            next_state=self.next_state_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
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