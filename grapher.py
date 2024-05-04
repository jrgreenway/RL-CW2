import matplotlib.pyplot as plt
from datetime import datetime
import os

class Grapher():
    def __init__(self, data:dict):
        self.data = data
        os.makedirs('graphs', exist_ok=True)
        self.plot_data()

    def plot_data(self):

        #timestamp for graphs
        nowtimestamp = datetime.now()
        timestamp = nowtimestamp.strftime("%m_%d,%H_%M_%S")

        # Plot each column against the index and save the graph
        for key in self.data.keys():
            plt.figure(figsize=(10, 6))
            plt.plot(self.data[key])
            plt.title(key.capitalize())
            plt.xlabel('Episode')
            plt.ylabel(key.capitalize())
            plt.grid(True)
            plt.savefig(f'graphs/{timestamp}{key}.png')
            plt.close()

