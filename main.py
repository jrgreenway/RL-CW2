import gymnasium as gym
from DeepQNetworkAgent import DQNAgent
from grapher import Grapher
import logging

env = gym.make("LunarLander-v2", continuous=False) 


agent = DQNAgent(env, 0.001, 64, 0.99, 0.3, 0.01, max_memory_size=10000)
total_frames = int(1e5) 
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

data, params = agent.train(total_frames)

grapher = Grapher(data, params)
logging.shutdown()