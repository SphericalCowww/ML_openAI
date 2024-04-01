import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.utils import play

######################################################################################################
if __name__ == "__main__":
    envName    = "LunarLander-v2"
    renderMode = "rgb_array_list"
    keyMap = {'j': 1,
              'k': 2,
              'l': 3}
    defaultAction = 0
    env = gym.make(envName, render_mode=renderMode)
    play.play(env, keys_to_action=keyMap, noop=defaultAction)
    env.close()

 
