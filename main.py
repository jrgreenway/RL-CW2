import gymnasium as gym 
from DeepQNetworkAgent import DQNAgent
from grapher import Grapher
import logging

env = gym.make("ALE/Tennis-ram-v5") # "ALE/Surround-v5"  #render_mode'human' ?

agent = DQNAgent(env, 0.0001, 64, 0.99, 0.3, 0.01, max_memory_size=10000)
total_frames = int(1e6) # Think we should consider changing to number of frames not episodes
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

data, params = agent.train(total_frames)

grapher = Grapher(data, params)
logging.shutdown()