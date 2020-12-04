#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:19:06 2020

@author: kerrick
"""


import pickle 

import os
    
'''
INSTRUCTIONS

run

    aws s3 cp s3://projects.irll/mc_group2/Trials/ ./ --recursive
    
from terminal to download trial data into current folder. In my case I ran
this command from basedir below. You should change basedir to the directory
where you downloaded the trial data.

Then just run this script.

'''
basedir = '/home/kerrick/uAlberta/RLI/Project/myCOACH/groupx_data/'
dirs = os.listdir(basedir)

data = []
for d in dirs:
    trial_data = []
    for file in os.listdir(basedir+d):
        episode_data = []
        with (open(basedir+d+'/'+file,'rb')) as openfile:
            while True:
                try:
                    episode_data.append(pickle.load(openfile))
                except EOFError:
                     break
        trial_data.append(episode_data)
    data.append(trial_data)


import numpy as np
person_data = []
person_budget_data = []
for person in data:
    episode_data = []
    episode_budget_data = []
    for episode in person:
        trial_data = []
        for trial in episode:
            try:
                trial_data.append(trial['steps'])
            except:
                print('no steps')
        episode_data.append(np.mean(trial_data))
        try:
            episode_budget_data.append(episode[0]['cumulative_budget_used'])
        except:
            print('no cumulative budget used')
    person_data.append(episode_data)
    person_budget_data.append(episode_budget_data)
    

import matplotlib.pyplot as plt

for p in range(len(person_data)):
    order = np.argsort(person_budget_data[p])
    plt.plot(np.array(person_data[p])[order])
    plt.title('Average Steps per Episode - Participant ' + str(p))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Steps')
    plt.show()

