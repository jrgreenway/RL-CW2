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
    epsilon=0.5,
    replace_target_nn=500,
    max_memory_size=100000)

total_frames = int(1e3) # Think we should consider changing to number of frames not episodes
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

data = agent.train(total_frames)
print(colored('Training Complete', 'green'))
grapher = Grapher(data)