import gymnasium as gym 
import numpy as np
import mujoco
from Agent import Agent
'''
I suggest you make a virtual enviroment in the code folder called .venv - you can do it on vscode
by putting '>Python: Select Interpreter' then pressing 'create virtual enviroment' then venv.
then in the terminal at the bottom (should say (.venv) before your path) do:
'pip install gymnasium[MuJoCo]', then 'pip uninstall mujoco', then 'pip install mujoco==2.3.7'
because the newest version doesnt work with gymnasium.

if it doesnt work try making a virtual enviroment in this folder called .venv (ask me if you need help)
the docs for the env -> https://gymnasium.farama.org/environments/mujoco/humanoid_standup/
'''

env = gym.make('HumanoidStandup-v4',render_mode = 'human')
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
