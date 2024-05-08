import gymnasium as gym 
from DeepQNetworkAgent import DQNAgent
from grapher import Grapher
from termcolor import colored
from gymnasium.wrappers import RecordVideo

total_frames = int(1e5)

def step_trigger(step):
    return step == 0 or (step > total_frames-1000)

def ep_trig(episode):
    return episode == 0 or (episode % 5 == 0 and episode > 250)

for iter in range(5):
    env = gym.make("LunarLander-v2", continuous=False, render_mode='rgb_array')
    env = RecordVideo(env, video_folder="videos/", episode_trigger=ep_trig, video_length=0)

    agent = DQNAgent(
        env,
        learning_rate=0.001,
        batch_size=64,
        gamma=0.99,
        beta=0.95,
        hidden_neurons=64,
        alpha=0.6,
        min_return_value=-500,
        max_return_value=300,
        replace_target_nn=1000,
        max_memory_size=100000,
        log=False)


    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    data, params = agent.train(total_frames)
    print(colored('Training Complete', 'green'))
    grapher = Grapher(data, params)