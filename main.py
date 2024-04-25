import gymnasium as gym 
import numpy as np
from DeepQNetworkAgent import DQNAgent
from matplotlib import pyplot as plt


env = gym.make("ALE/Surround-v5")#render_mode='human'  #Testing with Lunar Lander
action_space = env.action_space
print(action_space)
# print(env.metadata) -> 'render_modes': ['human', 'rgb_array', 'depth_arr
total_episodes = 100
agent = DQNAgent(learning_rate=0.003, batch_size=64, observation_space=[8], n_actions=4,
                hidden_neurons=128, max_memory_size=100000, epsilon=1.0, eps_decay=5e-4, eps_min=0.01, gamma=0.95)
rewards = agent.train(env, total_episodes)
plt.plot(rewards)
plt.show()
        
env.close()