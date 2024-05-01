'''
James - Suggest Using this file for all general utils too.
Found here: https://github.com/celsod/DDQN_Phil_Tabor/blob/master/utils.py
'''

'''
This is a program that was created by Phil Tabor.  I am using his program to
follow along in his tutorials.

YouTube: Machine Learning with Phil
Website: http://www.neuralnet.ai/
Twitter: https://twitter.com/mlwithphil

'''

import matplotlib.pyplot as plt
import numpy as np
import torch


def normalise(input:np.ndarray):
    '''Normalises Pixel inputs to be between 0 and 1'''
    return input / 255
    
def preprocess(input) -> np.ndarray:
    '''Preprocesses state observations'''
    state = np.array(input)
    state = normalise(state) # Normalises pixel values between 0 and 1
    state = np.expand_dims(state, axis=1) # Adds a channel dimension for CNN
    return state
    


def plotLearning(x, scores, epsilons, filename):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-5):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)