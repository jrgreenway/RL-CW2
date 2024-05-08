import gymnasium as gym 
from DeepQNetworkAgent import DQNAgent
from grapher import Grapher
from termcolor import colored


env = gym.make("LunarLander-v2", continuous=False)

agent = DQNAgent(
    env,
    learning_rate=0.0001,
    batch_size=64,
    gamma=0.99,
    min_return_value=0,
    max_return_value=500,
    replace_target_nn=500,
    max_memory_size=10000)

total_frames = int(1e4) # Think we should consider changing to number of frames not episodes
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

data = agent.train(total_frames)
print(colored('Training Complete', 'green'))
grapher = Grapher(data)