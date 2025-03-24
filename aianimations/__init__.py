"""
aianimations module

This package provides animation utilities for reinforcement learning
environments (CartPole and Pendulum) based on trained models.
"""

from .animate_cartpole import (
    get_frames as cartpole_get_frames,
    save_frames_as_gif as cartpole_save_gif,
    act_dqn as cartpole_act_dqn,
    act_qtable as cartpole_act_qtable,
    main as cartpole_main
)
from .animate_pendulum import (
    play as pendulum_play,
    save_frames_as_gif as pendulum_save_gif,
    main as pendulum_main,
    test as pendulum_test
)

__all__ = [
    "cartpole_get_frames", "cartpole_save_gif", "cartpole_act_dqn", "cartpole_act_qtable", "cartpole_main",
    "pendulum_play", "pendulum_save_gif", "pendulum_main", "pendulum_test"
]
