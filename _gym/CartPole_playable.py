import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
#import gym                  #https://www.gymlibrary.dev
#from gym.utils.play import play
import gymnasium as gym
from gymnasium.utils import play

######################################################################################################
if __name__ == "__main__":
    envName    = "CartPole-v1"
    renderMode = "rgb_array_list"
    keyMap = {'j': 0,
              'l': 1}
    env = gym.make(envName, render_mode=renderMode)
    play.play(env, keys_to_action=keyMap)
    
    env.close()

 
