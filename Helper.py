import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

class Helper():
    def __init__(self):
        self.data = pd.DataFrame()

    def add_data(self, data):
        self.data = self.data.append(data)

    def log_to_tensorboard(self):
        for i in range(len(self.data[0])):
            self.writer.add_scalar('Total Reward', self.data['Total Reward'][i], self.data['Episode'][i])
            self.writer.add_scalar('Epsilon', self.data['Epsilon'][i], self.data['Episode'][i])
            self.writer.add_scalar('Loss', self.data['Loss'][i], self.data['Episode'][i])

    def close_tensorboard(self):
        self.writer.close()

    
