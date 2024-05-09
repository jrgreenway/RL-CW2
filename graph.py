import matplotlib.pyplot as plt
import os
import json
import numpy as np

files = os.listdir("data/")

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size
    

    
    
for file in files:
    with open("data/"+file, "r") as f:
        data = json.load(f)
    data = data["data"]
    fig, axs = plt.subplots(nrows=1, ncols=len(data.keys()), dpi=600, figsize=(15,5))
    for i, key in enumerate(data.keys()):
        if key == "scores":
            x,y = range(len(data[key])),data[key]
            axs[i].scatter(x,y, label="Rewards", marker="x", )
            #moving_avg = moving_average(y, 20)
            #axs[i].plot(moving_avg, color='red')
            trend = np.polyfit(x, y, 3)
            trendline = np.poly1d(trend)
            axs[i].plot(x, trendline(x), color='green', label="Reward Trend")
            axs[i].set_title("Rewards")
            axs[i].set_ylabel("Reward")
            axs[i].set_xlabel("Episode")
            axs[i].grid(True)
            
        else:
            axs[i].plot(data[key])
            axs[i].set_title(key.capitalize())
            axs[i].set_ylabel(key.capitalize())
            axs[i].set_xlabel("Step")
        if len(axs[i].lines) >1:
            axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=len(axs[i].lines))
    plt.tight_layout()
    plt.savefig(f"{file}_plot.png")
    
    