import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import numpy as np
import random


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env = ImgObsWrapper(env)
    env.seed(seed)
    return env

def change_state(raw_state):
    modified_state = np.reshape(raw_state, (147,)) 
    return modified_state

def convert_reward(reward):
    if reward == 'good':
        r = 1
    elif reward == 'bad':
        r = -1
    elif reward == None:
        r = 0

    print('converted reward', r)
    if r!=0:
        update = True
    else:
        update = False
    
    return r, update

def normalize(v):
    if v[0][0] == 0 and v[0][1] == 0 and v[0][2] == 0 and v[0][3] == 0:
        return v
    normal_v = v / np.linalg.norm(v)
    return normal_v

def weighted_action(state, dqnagent, tameragent, HW, eps):
    EW = 1 - HW
    dqn_action_values = dqnagent.get_action_values(state)
    tamer_action_values = tameragent.get_action_values(state)
    normal_tamer_value = normalize(tamer_action_values)
    normal_q_value = normalize(dqn_action_values)
    print(normal_q_value, normal_tamer_value)
    weighted_average = HW*normal_tamer_value + EW*normal_q_value

   
    if random.random() > eps:
        return np.argmax(weighted_average.cpu().data.numpy())
    else:
        return random.choice(np.arange(tameragent.action_size))
    if index_of_action == tamer_action:
        tamer_action_used = 1
    else:
        tamer_action_used = 0
    return movement, index_of_action, tamer_action_used