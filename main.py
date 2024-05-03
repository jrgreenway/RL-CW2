import gymnasium as gym 
import numpy as np
from tqdm import tqdm
from DeepQNetworkAgent import DQNAgent
from matplotlib import pyplot as plt
from grapher import Grapher
from utils import plotLearning 
from termcolor import colored
import torch


env = gym.make("CartPole-v1") # "ALE/Surround-v5"  #render_mode'human' ?

agent = DQNAgent(env, 0.001, 64, 0.8, 1, replace=500, max_memory_size=10000)
total_episodes = 200 # Think we should consider changing to number of frames not episodes
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

data = agent.train(total_episodes)

grapher = Grapher(data)