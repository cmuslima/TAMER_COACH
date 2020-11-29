

import pickle
from scipy.io import savemat


files = []

for i in range(0,20):
    files.append("episode_" + str(i) + "_user_4c4f0575-078c-403e-8975-49d4a2c743ae") #this I just copied from one of the participants 
 #b/c I noticed the only change in the file name is the epsiode number   


objects = []
for file in files:
    with (open(file, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break



data=[]
for i in range(len(objects)):
    data.append((objects[i]['cumulative_budget_used'], objects[i]['dR'], objects[i]['steps'], i))
