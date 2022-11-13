import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
import gym                  #https://www.gymlibrary.dev
from gym.utils.play import play

######################################################################################################
if __name__ == "__main__":
    envName    = "LunarLander-v2"  #"LunarLander-v2", "CartPole-v1"
    renderMode = "human"        #"human", "rgb_array"
    randSeed   = 1

    env = gym.make(envName, render_mode=renderMode)#; play(env)

    env.action_space.seed(randSeed)
    observation, info = env.reset(seed=randSeed)
    for _ in range(1000):
        #action = policy(observation)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(action, observation, reward, terminated, truncated, info)
        if terminated or truncated: observation, info = env.reset()
    env.close()


 
