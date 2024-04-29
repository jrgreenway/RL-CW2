import pandas as pd
#pip install matplotlib if needed
import matplotlib.pyplot as plt
#import torch
#from torch.utils.tensorboard import SummaryWriter

class Helper():
    def __init__(self, log_dir='logs'):
        #init dataframe
        self.data = pd.DataFrame()
        #init logs for tensor
        self.writer = SummaryWriter(log_dir=log_dir)
        

    def add_data(self, data):
        #add data from agent
        self.data = self.data.append(data, ignore_index=True)

    def plot_data(self):
        #convert columns to tensor
        #for column in self.data.columns:
         #   tensor_data = torch.tensor(self.data[column].values)
         #   self.writer.add_scalar(column, tensor_data)

        #self.writer.close()
        #self.writer.add_scalar(0, data)
        #add later

        #makes the directory
        os.makedirs(Learning_Graphs, exist_ok=True)
        #timestamp for graphs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Plot each column against the index and save the graph
        for column in self.data.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(self.data.index, self.data[column])
            plt.title(column)
            plt.xlabel('Index')
            plt.ylabel(column)
            plt.grid(True)
            plt.savefig(f'{Learning_Graphs}{column}_{timestamp}.png')
            plt.close()
            print("test")

    #def close_tensorboard(self):
    #    self.writer.close()

helper = Helper()

# Adding some data (example)
data = {'Episode': [1, 2, 3],
        'Total Reward': [100, 120, 150],
        'Epsilon': [0.1, 0.08, 0.05],
        'Loss': [0.5, 0.4, 0.3]}
helper.add_data(data)

# Plotting the data
helper.plot_data()

    




    
