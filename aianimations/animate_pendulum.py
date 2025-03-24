"""
animate_pendulum.py
"""
import gym
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import collections
import os
try:
    import pendulum #Pendulum is an instance of DQN for pyTorch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Missing required DQN implementation for pyTorch')
import torch

global env
env = gym.make('Pendulum-v1')
input_dim = env.observation_space.shape[0]
output_dim=4
n_episode=300
max_episode=200
batch_size=16
min_eps=0.00175
hidden_dim = 90
gamma = 0.95


Frame = collections.namedtuple('Frame', ['image', 'episode', 'penalty'])
dqn = pendulum.DQN(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
try:
    dqn.load_state_dict(torch.load('./pendulumDQN.pt')) #pendulumDQN.pt is the pyTorch saved training data
except FileNotFoundError:
    raise FileNotFoundError('you do not have any training data in the current working directory')

dqn.eval()

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].image.shape[1] / 72.0, frames[0].image.shape[0] / 72.0), dpi=72)
    ax = plt.gcf().add_subplot()

    plt.axis('off')

    patch = plt.imshow(frames[0].image)
    at = ax.text(x=0.05, y=0.5, s=f"Episode: {frames[0].episode}\nPenalty: {frames[0].penalty}", fontsize=18 \
        , bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10}, fontweight='black', fontfamily='serif')

    def animate(i):
        patch.set_data(frames[i].image)
        at.set_text(s=f"Episode: {frames[i].episode}\nPenalty: {frames[i].penalty}")

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=100)
    anim.save(path + filename, writer='imagemagick', fps=30)

def play(actions:np.ndarray, getFrames:bool=False, episode:int=0):
    s = env.reset()
    done = False
    penalty = 0
    if getFrames:
        frames=[]
    while not done:
        if getFrames:
            frames.append(Frame(env.render(mode='rgb_array'), episode, penalty))
        else:
            env.render()

        #convert state as array of floats to Tensor
        state = torch.autograd.Variable(torch.Tensor(s.reshape(-1, input_dim)))

        #get the q-value (expected penalty scores) from the state
        #dqn.train(mode=False)
        scores = dqn(state)

        #get the analogue action from scores
        _, argmax = torch.max(scores.data, 1)
        a = int(argmax.numpy())

        s2, r, done, _ = env.step(np.ndarray((1,), buffer=np.array(actions[a])))
        penalty -= round(r, )
        s = s2
    if getFrames:
        return frames
    else:
        return penalty

def discretize_actions(n_actions, upper_bounds, lower_bounds):
    """
    Takes an observation of the environment and aliases it.
    By doing this, very similar observations can be treated
    as the same and it reduces the state space so that the
    Q-table can be smaller and more easily filled.

    Input:
    actions (int) : number of possible actions

    Output:
    discretized (List) : discrete list of possible actions
    """
    n_actions -= 1
    return np.arange(lower_bounds, \
        upper_bounds +((upper_bounds - lower_bounds) / n_actions) \
        , (upper_bounds - lower_bounds) / n_actions, dtype=float)

discrete_action_values = discretize_actions(output_dim, env.action_space.high, env.action_space.low)

def main():
    path='./'
    filename='gym_animation.gif'
    n = 0
    while filename in os.listdir(path):
        n += 1
        filename = 'gym_animation' + str(n) + '.gif'
    frames = []
    for i in range(1,4):
        frames += play(discrete_action_values, True, i)
    env.close()
    save_frames_as_gif(frames, filename=filename)

def test():
    print(discrete_action_values)
    print(play(discrete_action_values))

if __name__ == '__main__':
    main()
    #test()
