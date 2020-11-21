#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 22:27:49 2020

@author: kerrick

This will unpickle data (here named "saved_policy") saved from a trial and save the weights 
in a file (here called "trained_weights")

The data needs to have a 'w' key and it needs to be in the same folder 
you run this code from (else include the path when you open the file)

Weights will be saved in the same directory you run this file from

"""
import pickle
from scipy.io import savemat

objects = []
file = "saved_policy5"
with (open(file, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
weights = objects[0]['w']
savemat('trained_weights5',mdict={'w':weights})
