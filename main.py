import gymnasium as gym 
import numpy as np
from tqdm import tqdm
from DeepQNetworkAgent import DQNAgent
from matplotlib import pyplot as plt
from grapher import Grapher
from utils import plotLearning 
from termcolor import colored
import torch


env = gym.make("ALE/Surround-v5", obs_type='ram') # "ALE/Surround-v5"  #render_mode'human' ?

agent = DQNAgent(env, 0.003, 64, 0.5, 1)
total_episodes = 100
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

data = agent.train(total_episodes)

grapher = Grapher(data)