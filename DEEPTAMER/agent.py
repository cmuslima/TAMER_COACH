'''
This is a demo file to be replaced by the researcher as required.
This file is imported by trial.py and trial.py will call:
start()
step()
render()
reset()
close()
These functions are mandatory. This file contains minimum working versions 
of these functions, adapt as required for individual research goals.
'''
import gym
import gym_minigrid
from helper_functions import make_env, convert_reward, change_state, weighted_action
import random
import numpy as np
from deeptameragent import TamerAgent
from dqn_agent import DQNAgent
import time

class Agent():
    '''
    Use this class as a convenient place to store agent state.
    '''
    
    def start(self, game:str):
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
        seed = 0
        self.HW = .5
        self.eps = .01
        self.env = make_env(game, seed)
        self.tameragent= TamerAgent(state_size=147, action_size=6, seed=seed)
        self.dqnagent = DQNAgent(state_size=147, action_size=6, seed=seed)
        return self.env

    def update_agent_with_human_feedback(self,reward):
    
        reward, update = convert_reward(reward)

        self.tameragent.step(update, self.state, self.action, reward, self.next_state, self.done, self.time) #this says I will make an update if I get a feedback.

        if update:
            self.numfeedbacks+=1

        envState ={'S': self.state, 'A': self.action, 'R': reward, 'S prime': self.next_state, 'done': self.done, 'totalfeedbacks':self.numfeedbacks, 'win': self.win, 'numtimesteps': self.numtimesteps} 
      
        return envState
    def step(self, reward:str):
        '''
        Trajectory looks like: S, A, R, S'
        The human will provide feedback once the agent takes an action which leads to a subsequent next state, S'.
        Because we are using Deep Tamer, we must know S' and D (Done) along with the feedback to update the experience replay buffer.
        
        So we update (S, A, S') with the feedback.
        
        Then we set S as S' and take a new action.

        Caller: 
            - Trial.take_step()
        Inputs:
            - env (Type: OpenAI gym Environment)
            - human feedback (Type: string)
        Returns:
            - envState (Type: dict containing all information to be recorded for future use)
              change contents of dict as desired, but return must be type dict.
        '''
       
        print("REWARD", reward)
        if self.numtimesteps == 0:
            self.agent_start()

        envState = self.update_agent_with_human_feedback(reward) 
        if self.done:
            return envState, -1

        
        old_action = self.action


        self.state = self.next_state
        #self.action = self.tameragent.act(self.state, self.eps)
        self.action = weighted_action(self.state, self.dqnagent, self.tameragent, self.HW, self.eps)

        self.time = time.time()
        next_state, self.env_reward, self.done, _ = self.env.step(self.action)
        
        self.next_state = change_state(next_state)
        self.dqnagent.step(self.state, self.action,  self.env_reward, self.next_state, self.done)
        
        self.numtimesteps+=1
        if self.env_reward!= 0:
            self.win = True
        return envState, old_action

    def agent_start(self):
        self.time = time.time()
        next_state, self.env_reward, self.done, _ = self.env.step(self.action)
        self.next_state = change_state(next_state)
        self.dqnagent.step(self.state, self.action,  self.env_reward, self.next_state, self.done)

    def render(self):
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
        return self.env.render('rgb_array')
    
    def reset(self):
        '''
        Resets the environment to start new episode.
        Caller: 
            - Trial.reset()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns: 
            No Return
        '''
        self.win = False
        self.numfeedbacks = 0 
        self.numtimesteps = 0 
        self.state = change_state(self.env.reset()) #first state
        self.action_size = self.tameragent.action_size
        print('action size', self.tameragent.action_size)
        self.action = random.randint(0, self.action_size-1) #first action
        print('self.action', self.action)
        
    def close(self):
        '''
        Closes the environment at the end of the trial.
        Caller:
            - Trial.close()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns:
            No Return
        '''
        self.env.close()
