import gymnasium as gym 
import numpy as np
from DeepQNetworkAgent import DQNAgent
from matplotlib import pyplot as plt
from utils import plotLearning 
from termcolor import colored
import torch


env = gym.make("CartPole-v1") # "ALE/Surround-v5"  #render_mode'human' ?
agent = DQNAgent(learning_rate=5e-4, batch_size=64, observation_space=env.observation_space.shape, \
                  n_actions=env.action_space.n, epsilon=1.0, eps_decay=1e-3, eps_min=0.01, gamma=0.95, replace = 100)
scores, eps_history = [], []
total_episodes = 100
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

for i in range(total_episodes): # Loop through the episodes
    score = 0
    terminated = False
    truncated = False   # Truncated is for time limits when time is not part of the observation space / state
    (state,_) = env.reset()
    counter = 0
    while not terminated and not truncated:
        #env.render()
        #if counter % 100 == 0: print(colored(f"State Number: {counter}, Epsilon: {agent.epsilon}", "red"))
        counter += 1
        action = agent.action(state)                                        # Agent picks an action
        next_state, reward, terminated, truncated, info = env.step(action)  # Env returns next state, reward, and if it's done
        score += reward                                                     # Total score is updated
        agent.store_transition(state, action, reward, next_state, terminated, truncated) # Store the experience
        agent.learn()                                                       # Learns from the experience
        state = next_state                                                  # Update the state                                            
        if terminated or truncated: print(colored("Episode terminated or truncated", "red"))
    scores.append(score)                                     # Episode done. Append the score to the scores list
    eps_history.append(agent.epsilon)                      # Append the epsilon value to the eps_history list, for data analysis

    avg_score = np.mean(scores[-100:])                  # Average score over last 100 episodes

    print('episode ', i, 'score %.2f' % score,
            '  average score %.2f' % avg_score,
            ' epsilon %.2f' % agent.epsilon)
    
if i > 10 and i % 10 == 0:
    agent.save_models()

x = [i+1 for i in range(total_episodes)]
filename = 'surround.png'
plotLearning(x, scores, eps_history, filename)