import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

class Helper():
    def __init__(self):
        self.data = {'Episode': [],
                     'Total Reward': [],
                     'Epsilon': [],
                     'Loss': []}
        self.writer = SummaryWriter()

    def add_data(self, episode, total_reward, epsilon, loss):
        self.data['Episode'].append(episode)
        self.data['Total Reward'].append(total_reward)
        self.data['Epsilon'].append(epsilon)
        self.data['Loss'].append(loss)

    def log_to_tensorboard(self):
        for i in range(len(self.data['Episode'])):
            self.writer.add_scalar('Total Reward', self.data['Total Reward'][i], self.data['Episode'][i])
            self.writer.add_scalar('Epsilon', self.data['Epsilon'][i], self.data['Episode'][i])
            self.writer.add_scalar('Loss', self.data['Loss'][i], self.data['Episode'][i])

    def close_tensorboard(self):
        self.writer.close()

# Logging data to TensorBoard
helper.log_to_tensorboard()

# Close TensorBoard writer
helper.close_tensorboard()
    
