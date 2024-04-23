import gymnasium as gym 
import numpy as np
from Agent import Agent

env = gym.make("ALE/Surround-v5")#render_mode='human'
total_episodes = 10
agent = Agent(env.observation_space.shape, env.action_space.n)
rewards = agent.train(env, total_episodes)
        
env.close()