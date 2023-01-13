from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import BoxStyle
import gym
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import collections
import os

env = gym.make('CartPole-v1')
try:    
    Q_table = np.load('cartpoleData.npy') #cartpoleData is an npy 2d array file storing data from training
except FileNotFoundError:
    raise FileNotFoundError('you do not have any training data in the current working directory')
upper_bounds = [4.8, 0.5, 0.41887903, 0.8726646259971648]
lower_bounds = [-4.8, -0.5, -0.41887903, -0.8726646259971648]
buckets = (3, 3, 6, 6)

Frame = collections.namedtuple('Frame', ['image', 'action', 'reward'])
    
def getFrames():
    t = 0
    done = False
    current_state = discretize_state(env.reset())
    frames = []
    printAction = ['←', '→']
    while not done:
        t = t+1
        action = np.argmax(Q_table[current_state])
        frames.append(Frame(env.render(mode='rgb_array'), printAction[int(action)], t))
        obs, _, done, _ = env.step(action)
        new_state = discretize_state(obs)
        current_state = new_state
    env.close()
    return frames

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

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
    anim.save(path + filename, writer='imagemagick', fps=30)

def play():
    """Runs an episode while displaying the cartpole environment."""
    t = 0
    done = False
    current_state = discretize_state(env.reset())
    penalties = 0
    while not done:
        env.render()
        t = t+1
        action = np.argmax(Q_table[current_state])
        obs, reward, done, _ = env.step(action)
        new_state = discretize_state(obs)
        current_state = new_state
    print(f"Penalties: {penalties}")
    env.close()
    return t

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
                            in the buckets list.
    """
    discretized = list()
    for i in range(len(obs)):
        scaling = ((obs[i] + abs(lower_bounds[i])) 
                    / (upper_bounds[i] - lower_bounds[i]))
        new_obs = int(round((buckets[i] - 1) * scaling))
        new_obs = min(buckets[i] - 1, max(0, new_obs))
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