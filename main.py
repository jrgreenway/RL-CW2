import gymnasium as gym 
import numpy as np
from Agent import Agent
'''
you need to have done 'pip install gymnasium[MuJoCo]' in the terminal 
if it doesnt work try making a virtual enviroment in this folder called .venv (ask me if you need help)
the docs for the env -> https://gymnasium.farama.org/environments/mujoco/humanoid_standup/
'''

env = gym.make('HumanoidStandup-v4')
# print(env.metadata) -> 'render_modes': ['human', 'rgb_array', 'depth_array']
total_episodes = 100
agent = Agent(env)
for episode in range(total_episodes):
    observation, info = env.reset()
    terminated = False
    while not terminated:
        action = agent.choose_action()
        observation, reward, terminated, truncated, info = env.step(action)
        
env.close()
