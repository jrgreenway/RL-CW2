import gymnasium as gym 
from DeepQNetworkAgent import DQNAgent
from grapher import Grapher


env = gym.make("CartPole-v1") # "ALE/Surround-v5"  #render_mode'human' ?

agent = DQNAgent(env, 0.00001, 64, 0.95, max_return_value=500, min_return_value=0, max_memory_size=10000)
total_frames = int(1e3) # Think we should consider changing to number of frames not episodes
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

data = agent.train(total_frames)

grapher = Grapher(data)