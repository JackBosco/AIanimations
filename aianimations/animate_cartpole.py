"""
animate_cartpole.py
"""
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import BoxStyle
import gym
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import collections
import os

env = gym.make('CartPole-v1', render_mode='rgb_array')
Q_TABLE = None
DQN = None

UPPER_BOUNDS = [4.8, 0.5, 0.41887903, 0.8726646259971648]
LOWER_BOUNDS = [-4.8, -0.5, -0.41887903, -0.8726646259971648]
BUCKETS = (3, 3, 6, 6)

def load_qtable(path):
    global Q_TABLE
    try:
        Q_TABLE = np.load(path) #cartpoleData is an npy 2d array file storing data from training
    except FileNotFoundError:
        raise FileNotFoundError('you do not have any training data in the current working directory')

def load_dqn(model):
    global DQN
    DQN = model

Frame = collections.namedtuple('Frame', ['image', 'action', 'reward'])

def get_frames(max_episode=500):
    t = 0
    done = False
    state = (env.reset())
    state = np.array(state[0]) if len(state) != 4 else np.array(state)
    frames = []
    printAction = ['←', '→']
    while not done and t < max_episode:
        t = t+1
        if Q_TABLE:
            action = act_qtable(state)
        elif DQN:
            action = act_dqn(np.expand_dims(state, axis=0))
        else:
            raise ValueError("no qtable or dqn found. use load_dqn or load_qtable")
        frames.append(Frame(env.render(), printAction[int(action)], t))
        new_state, _, done, *_ = env.step(action)
        state = np.array(new_state)
    env.close()
    return frames

def save_frames_as_gif(frames, filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].image.shape[1] / 72.0, frames[0].image.shape[0] / 72.0), dpi=72)
    ax = plt.gcf().add_subplot()

    plt.axis('off')

    patch = plt.imshow(frames[0].image)
    at = ax.text(x=0.05, y=0.5, s=f"Action: {frames[0].action}\nReward: {frames[0].reward}", fontsize=18 \
        , bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10}, fontweight='black', fontfamily='serif')

    def animate(i):
        patch.set_data(frames[i].image)
        at.set_text(s=f"Action: {frames[i].action}\nReward: {frames[i].reward}")

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=100)
    anim.save(filename, writer='imagemagick', fps=30)

def act_qtable(state):
    """
    extracts single action from qtable given state using argmax
    requires quatble to be loaded
    """
    current_state = discretize_state(state)
    return np.argmax(Q_TABLE[current_state])

def act_dqn(state):
    """
    extracts single action from dqn given state using argmax
    requires dqn to be loaded
    """
    return np.argmax(DQN(state))

def discretize_state(obs):
    """
    Takes an observation of the environment and aliases it.
    By doing this, very similar observations can be treated
    as the same and it reduces the state space so that the
    Q-table can be smaller and more easily filled.

    Input:
    obs (tuple): Tuple containing 4 floats describing the current
                    state of the environment.

    Output:
    discretized (tuple): Tuple containing 4 non-negative integers smaller
                            than n where n is the number in the same position
                            in the BUCKETS list.
    """
    discretized = list()
    for i in range(len(obs)):
        scaling = ((obs[i] + abs(LOWER_BOUNDS[i]))
                    / (UPPER_BOUNDS[i] - LOWER_BOUNDS[i]))
        new_obs = int(round((BUCKETS[i] - 1) * scaling))
        new_obs = min(BUCKETS[i] - 1, max(0, new_obs))
        discretized.append(new_obs)
    return tuple(discretized)

def main():
    path='./'
    filename='gym_animation.gif'
    n = 0
    while filename in os.listdir(path):
        n += 1
        filename = 'gym_animation' + str(n) + '.gif'
    save_frames_as_gif(getFrames(), filename=filename)

if __name__ == '__main__':
    main()
