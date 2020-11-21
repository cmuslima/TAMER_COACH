#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:19:06 2020

@author: kerrick
"""
# import pandas as pd

# object = pd.read_pickle(r'testdata1')

# import pickle 

# objects = []
# with (open("testdata1", "rb")) as openfile:
    
#     objects.append(pickle.load(openfile, encoding='latin1'))
    
# import gzip

# with gzip.open("testdata1", 'rb') as ifp:
#     print(pickle.load(ifp))

# import _pickle as cPickle

import pickle 


content = pickle.load(open("testdata1.pickle", "rb"))

objects = []
# with (open("testdata1", "rb")) as openfile:
#     objects.append(pickle.Unpickler(openfile).load())
    
with (open("testdata1.pickle", "rb")) as openfile:
    
    objects.append(pickle.load(openfile))
  
    
with open('testdata1', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()
    print(p)


    
import pickle
import boto3

objects = []
# with (open("testdata1", "rb")) as openfile:
#     objects.append(pickle.Unpickler(openfile).load())
    
with (open("episode_1_user_46180150-a4ad-4c2f-802f-402cafdf35af", "rb")) as openfile:
    
    objects.append(pickle.load(openfile))


import numpy 

import pickle
with (open("episode_1_user_46180150-a4ad-4c2f-802f-402cafdf35af", "rb")) as f:
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    u.load()


with (open('testdata1', "rb")) as f:
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    u.load()
    
with (open("episode_5_user_46180150-a4ad-4c2f-802f-402cafdf35af", "rb")) as f:
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    u.load()

s3 = boto3.resource('s3')
my_pickle = pickle.loads(s3.Bucket("bucket_name").Object("key_to_pickle.pickle").get()['Body'].read())


import pickle
data = []
with (open('testdata2', "rb")) as f:
    u = pickle._Unpickler( f )
    #u.encoding = 'latin1'
    data.append(u.load())
    
import pickle
data = []
with (open('testdata2', "rb")) as f:
    data.append(pickle.load(f))
    data.append(pickle.load(f))
    
'''
Make sure to comment out lines 298-301 in trial.py (in the create_file function). See the lines below
        # if self.config.get('dataFile') == 'trial':
        #     self.outfile.write(f'User {self.userId}')
        # else:
        #     self.outfile.write(f'User {self.userId} Episode {self.episode}')

'''

import pickle
objects = []
with (open("saved_policy2", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
