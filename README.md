# AIanimations
python scripts to produce an informational gif from a sequence of environment renderings

### Cartpole no PyTorch

![cartpole](https://github.com/JackBosco/AIanimations/blob/main/cartpole.gif)

This creates a gif with the dynamic plot on the top right that can be configured.
You need to train cartpole yourself and provide a data file named cartpoleData.npy.
Since this solution does not use PyTorch, you only need to provide the data file
in the form of a NumPy matrix and don't need to implement any interface.

### Pendulum with PyTorch

![pendulum](https://github.com/JackBosco/AIanimations/blob/main/pendulum.gif)
This creates a gif with the dynamic plot on the top right that can be configured.
You need to train cartpole yourself and provide a data file named pendulumDQN.pt.
This solution uses PyTorch, so you must implement the DQN interface from PyTorch
as a requirement. The data file must also be compatible with the save() and load()
methods from PyTorch so that your DQN implementation can run from a data file.
