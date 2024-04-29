import pandas as pd
#pip install matplotlib if needed
import matplotlib.pyplot as plt
#import torch
#from torch.utils.tensorboard import SummaryWriter

#This currently produces, graphs for each column of data then overrides them if its run again

class Helper():
    def __init__(self, log_dir='logs'):
        #initialise dataframe
        self.data = pd.DataFrame()
        #to put into this function, pd.DataFrame(data) will convert a list data into a dataframe

        #init logs for tensor
        #self.writer = SummaryWriter(log_dir=log_dir)
        

    def add_data(self, data):
        #add data from agent
        new_data = pd.DataFrame.from_dict(data) #add more of these so no matter what form data is given in it can be appended
        #self.data = self.data.append(new_data) #ignore_index=True
        self.data = pd.concat([self.data, new_data], ignore_index=True)

    def plot_data(self):
        #convert columns to tensor
        #for column in self.data.columns:
         #   tensor_data = torch.tensor(self.data[column].values)
         #   self.writer.add_scalar(column, tensor_data)

        #self.writer.close()
        #self.writer.add_scalar(0, data)
        #add later

        output_dir = "Helper_graphs"

        #timestamp for graphs
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Plot each column against the index and save the graph
        for column in self.data.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(self.data.index, self.data[column])
            plt.title(column)
            plt.xlabel('Index')
            plt.ylabel(column)
            plt.grid(True)
            plt.savefig(f'{output_dir}_{column}.png')
            plt.close()

    #def close_tensorboard(self):
    #    self.writer.close()

#helper = Helper()

# Adding some data (example)
#data = {'Episode': [1, 2, 3, 4],
#        'Total Reward': [100, 120, 150, 5],
#        'Epsilon': [0.1, 0.08, 0.05, 5],
#        'Loss': [0.5, 0.4, 0.3, 5]}
#helper.add_data(data)

# Plotting the data
#helper.plot_data()
