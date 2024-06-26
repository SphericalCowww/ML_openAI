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
    renderMode = "human"
    randSeed   = 1

    env = gym.make(envName, render_mode=renderMode)
    env.action_space.seed(randSeed)
    observation, info = env.reset(seed=randSeed)
    while True:
        #action = policy(observation)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print("action:", action)
        print("observation:", observation)
        print("reward:", reward)
        print("terminated, truncated, info:", terminated, truncated, info)
        if terminated or truncated: observation, info = env.reset()
    env.close()

 
