import gymnasium as gym 
import numpy as np
from DeepQNetworkAgent import DQNAgent
from matplotlib import pyplot as plt
from utils import plotLearning 



env = gym.make("MountainCar-v0") # "ALE/Surround-v5"  #render_mode'human' ?
agent = DQNAgent(learning_rate=0.003, batch_size=64, observation_space=[2], n_actions=3, epsilon=1.0, eps_decay=5e-4, eps_min=0.01, gamma=0.95)
scores, eps_history = [], []
total_episodes = 10

for i in range(total_episodes): # Loop through the episodes
    score = 0
    terminated = False
    truncated = False   # Truncated is for time limits when time is not part of the observation space / state
    state = env.reset()
    counter = 0
    while not terminated and not truncated:
        if counter % 100 == 0: print(f"State: {counter}")
        counter += 1
        action = agent.action(state)                                        # Agent picks an action
        next_state, reward, terminated, truncated, info = env.step(action)  # Env returns next state, reward, and if it's done
        score += reward                                                     # Total score is updated
        agent.store_transition(state, action, reward, next_state, terminated, truncated) # Store the experience
        agent.learn()                                                       # Learns from the experience
        state = next_state                                                  # Update the state                                            
    scores.append(score)                                     # Episode done. Append the score to the scores list
    eps_history.append(agent.epsilon)                      # Append the epsilon value to the eps_history list, for data analysis

    avg_score = np.mean(scores[-100:])                  # Average score over last 100 episodes

    print('episode ', i, 'score %.2f' % score,
            'average score %.2f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
x = [i+1 for i in range(total_episodes)]
filename = 'lunar_lander.png'
plotLearning(x, scores, eps_history, filename)