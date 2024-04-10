import gymnasium as gym 
import numpy as np
from Agent import Agent


env = gym.make('HumanoidStandup-v4',render_mode = 'human')#change
# print(env.metadata) -> 'render_modes': ['human', 'rgb_array', 'depth_array']
total_episodes = 100
agent = Agent(env)
for episode in range(total_episodes):
    observation, info = env.reset()
    terminated = False
    while not terminated:
        env.render()
        action = np.full((17,), 0.2)#agent.choose_action()
        observation, reward, terminated, truncated, info = env.step(action)
        
env.close()
