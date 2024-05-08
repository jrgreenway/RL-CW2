import matplotlib.pyplot as plt
import os
import json
import numpy as np

files = os.listdir("data/")

rewards = None
losses = None
epsilons = None
labels = None
def check_none(var):
    return var==None
    

for file in files:
    with open(f"data/{file}", "r") as json_file:
        data = json.load(json_file)
    rew = np.array(data.get('rewards', None))
    if check_none(rewards) and (data.get('rewards', None) is not None):
        rewards = rew
    else:
        rewards.append(rew)
        
    loss = np.array(data.get('losses', None))
    if check_none(losses) and (data.get('losses', None) is not None):
        losses = loss
    else:
        losses.append(loss)
    
    epsilon = np.array(data.get('epsilons', None))
    if check_none(epsilons) and (data.get('epsilons', None) is not None):
        epsilons = epsilon
    else:
        epsilons.append(epsilon)
    
    label = data.get('label', None)
    if check_none(labels) and (data.get('label', None) is not None):
        labels = label
    else:
        labels.append(label)
        

rewards = np.array([[labels[i], rewards[i]] for i in range(len(labels))])
losses = np.array([[labels[i], losses[i]] for i in range(len(labels))])
epsilons = np.array([[labels[i], epsilons[i]] for i in range(min(len(labels), len(epsilons)))])

titles = ['rewards', 'losses', 'epsilons']
for title, dataset in zip(titles, (rewards, losses, epsilons)):
    fig, ax = plt.subplots()
    for run in dataset:
        ax.plot(run[1], label=run[0])
    ax.legend()
    plt.savefig(f"{title}.png", dpi=300)
        
        
    
        
