# Rainbow DQN for Lunar Lander

We (Group 24) have implemented rainbow DQN with the following modifications:
- Double Deep Q Network
- Prioritised Experience Replay (PER) (implementation aided by [1])
- Dueling Neural Networks
- Noisy Networks
- Categorical DQN
- n-step Learning

This entire implementation was helped along by [2], [3] and [4], with Prioritised Expereince Replay utilising the Segment Tree datastructure from [1]

## Requirements
To run this code, there are several requirements, and we suggest using a virtual environment.
We have developed and run the code using Python 3.11.9
Please use the command to download all the packages:
`pip install -r requirements.txt`
Or you may view the requirements.txt text file to view every library we have used.
To load the enviroment, you also need to have Swig installed on your computer.

## Running The Code
Simply run 'main.py', or ensure you are in the correct directory and input `python main.py` into the terminal. The Main branch of this repository represents the entire Rainbow DQN implementation, however we have also implemented several other variations:
- Normal DQN (on branch normal-dqn)
- Rainbow DQN without PER (on branch no-per)

We also have a simple graphing tool on branch graphing-tool.

To see logs of the training, pass `log=True` into DQNAgent in main.py. The code also records a video of the enviroment at the start, and every 5 episodes near the end of the training run, to disable this, comment out line 17 of main. Similarly, to not graph or save any data, comment out the grapher variable on line 40 .

## Citations

1. segment_tree.py has been derived from [OpenAI's baseline library](https://github.com/openai/baselines) for use in Prioritised Experience Replay (https://github.com/openai/baselines)

2. Some of DeepQNetworkAgent.py was made with the help of [Deep Q Learning is Simple with PyTorch | Full Tutorial 2020 by "Machine Learning with Phil"](https://www.youtube.com/watch?v=wc-FxNENg9U)(https://www.youtube.com/watch?v=wc-FxNENg9U)

3. Theoretical Background tied to code [rainbow-is-all-you-need GitHub repository](https://github.com/Curt-Park/rainbow-is-all-you-need)
(https://github.com/Curt-Park/rainbow-is-all-you-need)

4. Noisy Network implementation [NoisyNet A3C GitHub repositroy](https://github.com/Kaixhin/NoisyNet-A3C/tree/master)
(https://github.com/Kaixhin/NoisyNet-A3C/tree/master)