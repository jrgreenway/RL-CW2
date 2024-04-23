import gymnasium as gym 
import numpy as np
from Agent import Agent
from matplotlib import pyplot as plt


env = gym.make("ALE/Surround-v5")#render_mode='human'
action_space = env.action_space
print(action_space)
# print(env.metadata) -> 'render_modes': ['human', 'rgb_array', 'depth_arr
total_episodes = 10
agent = Agent()
rewards = agent.train(env, total_episodes)
plt.plot(rewards)
plt.show()
        
env.close()