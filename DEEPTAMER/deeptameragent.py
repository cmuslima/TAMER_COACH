#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday May 17 2021

@author: kerrick, callie
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time

from tamermodel import HNetwork
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
local_memory_batch_size = 10 
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate

UPDATE_EVERY = 5  # UPDATE FREQUENCY: how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TamerAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        assert self.state_size == 147
        self.seed = random.seed(seed)
        self.mintime = .28
        self.maxtime = 4

        # Q-Network
        self.hnetwork_local = HNetwork(state_size, action_size, seed).to(device)
        self.hnetwork_target = HNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.hnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        #self.localmemory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.localmemory =None
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def get_recent_experiences(self, ub, lb):
        print('getting recent experiences')
        self.localmemory = ReplayBuffer(self.action_size, BUFFER_SIZE, local_memory_batch_size, self.seed)
        for e in reversed(self.memory.memory):
           if e.time < ub and e.time > lb:
               self.add_to_local_memory(e.state, e.action, e.reward, e.next_state, e.done, e.time)
           if e.time <lb:
                break
 
    def add_to_local_memory(self, state, action, reward, next_state, done, time):
        # Save experience in replay memory
        self.localmemory.add(state, action, reward, next_state, done, time)
        #we only add to the memory buffer if we received an actual feedbac


    def add_to_memory(self, state, action, reward, next_state, done, time):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, time)
        #we only add to the memory buffer if we received an actual feedbac


    def get_time(self):
        current_time = time.time()
        lower_bound = current_time - self.maxtime 
        upper_bound = current_time - self.mintime
        return lower_bound, upper_bound, current_time

    def step(self, feedback_given, state, action, reward, next_state, done, time):

        self.add_to_memory(state, action, reward, next_state, done, time)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        lb, ub, _ = self.get_time()
        if feedback_given:
            self.get_recent_experiences(ub, lb)
            if len(self.localmemory) > local_memory_batch_size:
                experiences = self.localmemory.sample()
                self.learn(experiences, GAMMA)
                print('updating from recent feedback')

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn 
            #this looks like mini batch 
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.hnetwork_local.eval()
        with torch.no_grad():
            action_values = self.hnetwork_local(state)

        self.hnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def get_action_values(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.hnetwork_local.eval()
        with torch.no_grad():
            action_values = self.hnetwork_local(state)
        return action_values

    def evaluate_q(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.hnetwork_local.eval()
        with torch.no_grad():
            action_values = self.hnetwork_local(state)

        return action_values.numpy()[0]
    


    def learn(self, experiences, gamma): 
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, times = experiences

        #need to perform mini batch gradient descent  
        # get targets
        self.hnetwork_target.eval()
        with torch.no_grad():
            Q_targets_next = torch.max(self.hnetwork_target.forward(next_states), dim=1, keepdim=True)[0]

        Q_targets = rewards 
    
        # get outputs
        self.hnetwork_local.train()

        #Gathers values along an axis specified by dim.
        Q_expected = self.hnetwork_local.forward(states).gather(1, actions)

        #print(' Q_expected', Q_expected)
        #print('size of  Q_expected', np.shape(Q_expected))
        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # clear gradients
        self.optimizer.zero_grad()

        # update weights local network
        loss.backward()

        # take one SGD step
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.hnetwork_local, self.hnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
