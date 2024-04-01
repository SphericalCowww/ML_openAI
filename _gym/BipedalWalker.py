import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecVideoRecorder

######################################################################################################
if __name__ == "__main__":
    dir_name = "./output/"
    env_name = "BipedalWalker-v3"

    #virtual_display = Display(visible=0, size=(1400, 900))
    #virtual_display.start()
    env = gym.make(env_name, hardcore=True)
    env.reset()

    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", env.observation_space.shape)
 
    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", env.action_space.shape)
 
    env = make_vec_env('BipedalWalker-v3', n_envs=16)
    model = PPO(policy='MlpPolicy', env=env, n_steps=2048, batch_size=128, n_epochs=6, gamma=0.999,\
                gae_lambda=0.98, ent_coef=0.01, verbose=1)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for i in range(0, 20000, 1000):
        model.learn(total_timesteps=1000)
        model_name = dir_name + "ppo-" + env_name
        model.save(model_name)
        '''
        video_name = "replay_" + str(i) + ".mp4"
        generate_replay(model=model, video_length=100, is_deterministic=True,\
                        eval_env=DummyVecEnv([lambda: Monitor(gym.make(env_id, hardcore=True, 
                                                                       render_mode="rgb_array"))]),\
                        local_path=os.path.join(video_dir, video_name))
        '''
    print('Done')



