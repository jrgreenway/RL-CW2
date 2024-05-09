import matplotlib.pyplot as plt
import os
import json
import numpy as np

files = os.listdir("data/")

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size
    
scores = []
colours = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink', 'yellow', 'brown', 'cyan']    

for i, file in enumerate(files):
    with open("data/"+file, "r") as f:
        data = json.load(f)
    data = data["data"]
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=600, figsize=(8,6))
    for key in data.keys():
        if key == "scores":
            x,y = range(len(data[key])),data[key]
            #axs[i].scatter(x,y, label="Rewards", marker="x")
            moving_avg = moving_average(y, 20)
            scores.append((file[:-5], moving_avg))
            axs.plot(moving_avg, color=colours[i])

            #trend = np.polyfit(x, y, 3)
            #trendline = np.poly1d(trend)
            #axs[i].plot(x, trendline(x), color='green', label="Reward Trend")
            axs.set_title("Rewards")
            axs.set_ylabel("Reward")
            axs.set_xlabel("Episode")
            axs.grid(True)
        # elif key == "epsilons":
        #     axs[1].plot(data[key])
        #     axs[1].set_title(key.capitalize())
        #     axs[1].set_ylabel(key.capitalize())
        #     axs[1].set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig(f"{file}_plot.png")
    
fig, axs = plt.subplots(nrows=1, ncols=1, dpi=600, figsize=(8,6))
for name, line in scores:
    if name == "RANDOM":
        continue
    axs.plot(range(len(line)), line, label=name)
axs.set_ylabel("Reward")
axs.set_xlabel("Episode")
axs.grid(True)
axs.legend(loc='upper center', bbox_to_anchor=(0.5,1.1), ncol=len(scores)-1)
fig.savefig("ALL.png")
#