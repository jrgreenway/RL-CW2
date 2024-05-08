import gymnasium as gym 
from DeepQNetworkAgent import DQNAgent
from grapher import Grapher
from termcolor import colored


env = gym.make("CartPole-v1") # "ALE/Surround-v5"  #render_mode'human' ?

agent = DQNAgent(
    env,
    learning_rate=0.0001,
    batch_size=64,
    gamma=0.95,
    min_return_value=0,
    max_return_value=500,
    replace_target_nn=500,
    max_memory_size=10000)

total_frames = int(1e5) # Think we should consider changing to number of frames not episodes
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

data = agent.train(total_frames)
print(colored('Training Complete', 'green'))
grapher = Grapher(data)