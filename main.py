import gymnasium as gym 
import numpy as np
from Agent import Agent


env = gym.make("ALE/Surround-v5", render_mode='human')
action_space = env.action_space
print(action_space)
# print(env.metadata) -> 'render_modes': ['human', 'rgb_array', 'depth_arr
total_episodes = 100
agent = Agent(env)
for episode in range(total_episodes):
    observation, info = env.reset()
    terminated = False
    while not terminated:
        env.render()
        action = env.action_space.sample()#agent.choose_action()
        observation, reward, terminated, truncated, info = env.step(action)
        
env.close()