import gymnasium as gym 
import numpy as np
from DeepQNetworkAgent import DQNAgent
from matplotlib import pyplot as plt


env = gym.make("ALE/Surround-v5")#render_mode='human'
action_space = env.action_space
print(action_space)
# print(env.metadata) -> 'render_modes': ['human', 'rgb_array', 'depth_arr
total_episodes = 100
agent = DQNAgent()
rewards = agent.train(env, total_episodes)
plt.plot(rewards)
plt.show()
        
env.close()