#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:19:06 2020

@author: kerrick
"""


import pickle 


    
'''
Make sure to comment out lines 298-301 in trial.py (in the create_file function). See the lines below
        # if self.config.get('dataFile') == 'trial':
        #     self.outfile.write(f'User {self.userId}')
        # else:
        #     self.outfile.write(f'User {self.userId} Episode {self.episode}')

'''


objects = []
with (open("saved_policy2", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break