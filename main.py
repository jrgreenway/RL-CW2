import gymnasium as gym 
import numpy as np
from DeepQNetworkAgent import DQNAgent
from matplotlib import pyplot as plt
from utils import plotLearning 


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

def episode(self, env):
    state, info = env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        action = self.action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        #Put Experience Replay Here
        state = next_state
        total_reward += reward
    #self.update_target()
    return total_reward


env = gym.make('LunarLander-v2') # "ALE/Surround-v5"  #render_mode'human' ?
agent = DQNAgent(learning_rate=0.003, batch_size=64, observation_space=[8], n_actions=4, epsilon=1.0, eps_decay=5e-4, eps_min=0.01, gamma=0.95)
scores, eps_history = [], []
total_episodes = 500

for i in range(total_episodes): # Loop through the episodes
    score = 0
    terminated = False
    truncated = False   # Truncated is for time limits when time is not part of the observation space / state
    state = env.reset()
    while not terminated and not truncated:
        action = agent.action(state)                                        # Agent picks an action
        next_state, reward, terminated, truncated, info = env.step(action)  # Env returns next state, reward, and if it's done
        score += reward                                                     # Total score is updated
        agent.store_transition(state, action, reward, next_state, terminated, truncated) # Store the experience
        agent.learn()                                                       # Learns from the experience
        state = next_state                                                  # Update the state                                            
    scores.append(score)                                     # Episode done. Append the score to the scores list
    eps_history.append(agent.epsilon)                      # Append the epsilon value to the eps_history list, for data analysis

    avg_score = np.mean(scores[-100:])

    print('episode ', i, 'score %.2f' % score,
            'average score %.2f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
x = [i+1 for i in range(total_episodes)]
filename = 'lunar_lander.png'
plotLearning(x, scores, eps_history, filename)