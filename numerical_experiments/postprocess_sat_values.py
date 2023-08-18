
# coding: utf-8

# In[6]:

import pandas as pd
import os


# In[8]:

os.chdir("initial_conditions")


# In[9]:

with open("../sensors_of_interest.txt", 'r') as f:
    sensors_of_interest = f.read().split('\n')[:-1]


# In[12]:

with open("sat_observations.txt", 'w') as f:
    for sensor in sensors_of_interest:
        with open("flume_initialo.observation_well_flow.{}.dat".format(sensor.replace("_", "-"))) as g:
            final_line = g.readlines()[-1]
            sat = final_line.split()[2]
            f.write("{}\n".format(sat))

