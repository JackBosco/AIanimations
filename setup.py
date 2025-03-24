from setuptools import setup, find_packages

setup(
    name="aianimations",
    version="0.1.0",
    description="Animation utilities for CartPole and Pendulum environments using trained RL models.",
    author="Jack Bosco",
    url="https://github.com/JackBosco/AIAnimations",
    packages=find_packages(),
    install_requires=[
        "gym",
        "numpy",
        "matplotlib",
        "torch",
        # Note: Ensure that the module 'pendulum' (the PyTorch DQN implementation) is available
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)

