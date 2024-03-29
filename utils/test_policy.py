#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 22:08:36 2020

@author: kerrick
"""



import gym
import time
import numpy as np
import itertools


#This is the code for tile coding features
basehash = hash

class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval                        
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" +                " size:" + str(self.size) +                " overfullCount:" + str(self.overfullCount) +                " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)
    
    def fullp (self):
        return len(self.dictionary) >= self.size
    
    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates

from math import floor, log
from itertools import zip_longest

def tiles (ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

def tileswrap (ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b%numtilings) // numtilings
            coords.append(c%width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles


class MountainCarTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        """
            Initializes the MountainCar Tile Coder
            
            iht_size -- int, the size of the index hash table, typically a power of 2
            num_tilings -- int, the number of tilings
            num_tiles -- int, the number of tiles. Here both the width and height of the
            tile coder are the same
            """
        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
            
    def get_tiles(self, position, velocity):
        """
            Takes in a position and velocity from the mountaincar environment
            and returns a numpy array of active tiles.
            
            returns:
            tiles - np.array, active tiles
            """
        # Use the ranges above and self.num_tiles to scale position and velocity to the range [0, 1]
        # then multiply that range with self.num_tiles so it scales from [0, num_tiles]
        minP=-1.2
        maxP=.5
        minV=-.07
        maxV=.07
        scaleP= maxP- minP
        scaleV= maxV-minV
        
        position_scaled = ((position-minP)/(scaleP))*self.num_tiles
        
        velocity_scaled = ((velocity-minV)/(scaleV))*self.num_tiles
        
        
        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        # nothing to implment here
        mytiles = tiles(self.iht, self.num_tilings, [position_scaled, velocity_scaled])
        
        return np.array(mytiles)


#this is the coach agent class

class CoachAgent:
    """
        Initialization of Tamer Agent. All values are set to None so they can
        be initialized in the agent_init method.
        """
    def __init__(self):
        self.last_action = None
        self.previous_tiles = None
        self.first_state= None
        self.current_action = None
        self.current_tiles= None
        
        self.num_tilings =  8
        self.num_tiles =  8
        self.iht_size =  4096
        self.epsilon = 0.1
        self.gamma = 0 # this is discount
        self.trace_decay = 0.9 # trace decay parameter (lambda in COACH paper)
        self.x = 0.12 # was 0.08
        self.alpha =self.x/self.num_tilings  #this is step size
        self.initial_weights =  0.0
        self.num_actions = 3
        self.actions = list(range(self.num_actions))
        self.time_step=0
        self.experiences= list()
        self.max_n_experiences=1000
        self.window_size = 2
        self.feedback_delay = 0.6
        self.timestamp = time.time()

        
        # We initialize self.w to three times the iht_size. Recall this is because
        # we need to have one set of weights for each action.
        self.w = np.ones((self.num_actions, self.iht_size))
        
        
        # intialize trace to be same size as w
        self.trace = np.zeros(self.w.shape)
        
        self.softmax_prob = [0,0,0]
        
        # We initialize self.mctc to the mountaincar verions of the
        # tile coder that we created
        
        self.mctc = MountainCarTileCoder(iht_size=self.iht_size,
                                         num_tilings=self.num_tilings,
                                         num_tiles=self.num_tiles)
    

    def calculate_action_preferences(self, tiles):
        preferences = []    
        for a in range(self.num_actions):
            preferences.append(np.sum(self.w[a][tiles]))

        return preferences

    def gradient_logsoftmax(self, chosen_a, softmax_prob):   
        gradients = np.zeros(self.w.shape[0])
        for a in self.actions:
            if a == chosen_a:
                gradients[a] = (1 - softmax_prob[a])
            else:
                gradients[a] = -softmax_prob[a]
        
        return gradients

  

    def softmax_action_selection(self, state):
        
        position, velocity = state
        active_tiles=self.mctc.get_tiles(position, velocity)
        
        preferences = self.calculate_action_preferences(active_tiles)
        c = np.max(preferences)
        numerator = np.exp(preferences - c)
        denominator = np.sum(numerator)
        softmax_prob = np.array(numerator/denominator)
        chosen_action = np.random.choice(self.actions, p=softmax_prob)
        
        self.current_action = chosen_action
        self.current_tiles = np.copy(active_tiles)
        
        self.softmax_prob = softmax_prob
        
        return chosen_action
    
    def agent_start(self, state):
        """The first method called when the experiment starts, called after
            the environment starts.
            Args:
            state (Numpy array): the state observation from the
            environment's evn_start function.
            Returns:
            The first action the agent takes.
            """
        position, velocity = state
        
        active_tiles=self.mctc.get_tiles(position, velocity)
        
        self.current_action = np.random.choice(self.actions)
        self.current_tiles= np.copy(active_tiles)
        
        self.experiences.append((self.current_action, self.current_tiles, time.time()))
        return self.current_action

    def update_trace(self, active_tiles, grad):
        for a in self.actions:
            self.trace[a,active_tiles] = self.trace[a,active_tiles]*self.trace_decay + grad[a] 
        

    def update_policy(self, reward):
            
                
        if reward == 'good':
            r = 1
        elif reward == 'bad':
            r = -1
        elif reward == 'None':
            r = 0
        
        # First get state-action pair to be assigned credit
        current_time = time.time()
        
        i = 0                           # index for experience
        dt = self.feedback_delay + 1      # arbitrary value > self.feedback_delay
        
        if len(self.experiences) > 0:

            while dt > self.feedback_delay:
                expr = self.experiences[i] 
                dt = current_time - expr[2]  # index 2 of experience holds the timestamp
                i += 1    
                if i > len(self.experiences)-1:
                    break
            i -= 1 # decrement i so that it corresponds to expr
        else:
            return  # the experience buffer is empty; no update is possible
        
        # now expr holds the right state-action pair for the update
        
        self.experiences = self.experiences[i:]   # remove old experiences from buffer
        
        preferences = self.calculate_action_preferences(expr[1])
        
        #softmax_prob = np.exp(action_preferences)/np.sum(np.exp(action_preferences))
        c = np.max(preferences)
        numerator = np.exp(preferences - c)
        denominator = np.sum(numerator)
        softmax_prob = numerator/denominator
        
        grad = self.gradient_logsoftmax(expr[0], softmax_prob)
        self.update_trace(expr[1],grad) 
        
        # update w
        self.w += self.alpha*r*self.trace


   

#HIPPOGYM

def start(game:str):
    '''
        Starts an OpenAI gym environment.
        Caller:
        - Trial.start()
        Inputs:
        -   game (Type: str corresponding to allowable gym environments)
        Returs:
        - env (Type: OpenAI gym Environment as returned by gym.make())
        Mandatory
    '''
    np.random.seed(0)
    global myagent 
    myagent= CoachAgent()
    env = gym.make(game)
    
    return env

def step(env, reward:str):
    ##action:int is updated to reward: string
    '''
        Takes a game step.
        Caller:
        - Trial.take_step()
        Inputs:
        - env (Type: OpenAI gym Environment)
        - reward from the user (Type: string)
        Returns:
        - envState (Type: dict containing all information to be recorded for future use)
        change contents of dict as desired, but return must be type dict.
        '''

    if myagent.time_step == 0:
        myagent.agent_start(myagent.first_state)
        
    myagent.time_step += 1
    #myagent.timestamp = time.time()
        
    myagent.update_policy(reward)
    
    myagent.last_action= myagent.current_action
    myagent.previous_tiles= myagent.current_tiles
    
    state, _ , done, info = env.step(myagent.current_action)
    #envState = {'observation': state, 'done': done, 'action': myagent.current_action, 'policy': myagent.softmax_prob}
    envState = {'observation': state, 'done': 0, 'action': myagent.current_action, 'policy': myagent.softmax_prob,'reward':reward, 'w':myagent.w}
    #envState = {'observation': state, 'done': 0, 'action': myagent.current_action, 'timestep':myagent.time_step, 'timestamp':myagent.timestamp, 'experiences':myagent.experiences, 'w':myagent.w, 'trace':myagent.trace, 'reward':reward}
    # if done:

    if state[0] >= .5:
        envState['done']= True
        return envState
    
    else:
        envState['done']= False    
    
    myagent.softmax_action_selection(state)
    
    myagent.experiences.append((myagent.current_action, myagent.current_tiles, time.time()))
    
#print('len',len(myagent.experiences))

    return envState

def render(env):
    '''
        Gets render from gym.
        Caller:
        - Trial.get_render()
        Inputs:
        - env (Type: OpenAI gym Environment)
        Returns:
        - return from env.render('rgb_array') (Type: npArray)
        must return the unchanged rgb_array
        '''
    return env.render('rgb_array')

def reset(env):
    '''
        Resets the environment to start new episode.
        Caller:
        - Trial.reset()
        Inputs:
        - env (Type: OpenAI gym Environment)
        Returns:
        No Return
        '''
    myagent.first_state = env.reset()
    myagent.experiences= list()
    myagent.trace = np.zeros(myagent.w.shape)

def close(env):
    '''
        Closes the environment at the end of the trial.
        Caller:
        - Trial.close()
        Inputs:
        - env (Type: OpenAI gym Environment)
        Returns:
        No Return
        '''
    env.close()



'''
Code to load pre-trained weights and evaluate performance

'''
from scipy.io import loadmat
env = start('MountainCar-v0')
reset(env)
weight_file = '/home/kerrick/uAlberta/RLI/Project/myCOACH/Data/trained_weights5'  # change this to your weight file
myagent.w = loadmat(weight_file)['w']
for i in range(8000):
    envState = step(env,'None')
    env.render()
    state = envState['observation']
    if state[0] >= .5:
        envState['done']= True
        reset(env)
close(env)



