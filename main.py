import gymnasium as gym
from DeepQNetworkAgent import DQNAgent
from grapher import Grapher
import logging
from gymnasium.wrappers import RecordVideo

total_frames = int(1e5)

def step_trigger(step):
    return step == 0 or (step > total_frames-1000)

def ep_trig(episode):
    return episode == 0 or (episode % 5 == 0 and episode > 250)

for iter in range(5):
    env = gym.make("LunarLander-v2", continuous=False, render_mode='rgb_array')
    env = RecordVideo(env, video_folder="videos/", episode_trigger=ep_trig, video_length=0, name_prefix=f"run_{iter}")

    agent = DQNAgent(env,
                    0.001,
                    64,
                    0.99,
                    0.3,
                    0.01,
                    max_memory_size=10000,
                    hidden_neurons=64,
                    log=False)

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    data, params = agent.train(total_frames)

    grapher = Grapher(data, params)

logging.shutdown()