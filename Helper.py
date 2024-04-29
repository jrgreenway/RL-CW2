import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

class Helper():
    def __init__(self):
        #init dataframe
        self.data = pd.DataFrame()
        

    def add_data(self, data):
        #add data from agent
        self.data = self.data.append(data)

    def plot_data(self):
        #convert columns to tensor
        for column in self.data.columns:
            tensor_data = torch.tensor(self.data[column].values)
            self.writer.add_scalar(column, tensor_data)

        self.writer.close()

    def close_tensorboard(self):
        self.writer.close()

    
