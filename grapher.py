import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

class Grapher():
    def __init__(self, data:dict, args:dict):
        nowtimestamp = datetime.now()
        self.timestamp = nowtimestamp.strftime("%m_%d,%H_%M_%S")
        self.data = data
        self.args = args
        os.makedirs('graphs', exist_ok=True)
        self.plot_data()
        self.save_args()

    def save_all(self):
        to_save = dict(
            time=self.timestamp,
            parameters=self.args,
            data=self.data 
        )
        with open(f'graphs/{self.timestamp}_parameters.json', 'w') as f:
            json.dump(to_save, f)

    def plot_data(self):
        for key in self.data.keys():
            plt.figure(figsize=(10, 6))
            plt.plot(self.data[key])
            plt.title(key.capitalize())
            plt.xlabel('Episode')
            plt.ylabel(key.capitalize())
            plt.grid(True)
            plt.savefig(f'graphs/{self.timestamp}{key}.png')
            plt.close()
