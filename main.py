import gymnasium as gym 
import numpy as np
from tqdm import tqdm
from DeepQNetworkAgent import DQNAgent
from matplotlib import pyplot as plt
from grapher import Grapher
from utils import plotLearning 
from termcolor import colored
import torch


env = gym.make("CartPole-v1", render_mode='rgb_array') # "ALE/Surround-v5"  #render_mode'human' ?

agent = DQNAgent(env, 0.003, 64, 0.95, 1)
total_episodes = 250
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

data = agent.train(total_episodes)

grapher = Grapher(data)