import pandas as pd
import numpy as np
import datetime

flume_length = 8
horizontal_cells = 80

def get_header(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "variables" in line.lower():
                return lines[1].replace('"', '').split('=')[1].split(',')

def get_hydrograph_data(fname):
    return pd.read_csv(fname, skiprows=[0, 1, 2], delim_whitespace=True,
                header=None, index_col=0, names=get_header(fname),
                usecols=[0, 1])

stream_hydrographs = pd.DataFrame()
for index, x in enumerate(np.linspace(0.1, flume_length, horizontal_cells)):
    stream_hydrographs.loc[:, str(x)] = get_hydrograph_data('flumeo.hydrograph._x{}_.dat'.format(x))['Surface']            
    
toe_loc = []                           
for row in stream_hydrographs.iterrows():
    toe_loc_temp = 0.0
    for col in row[1].keys():
        if row[1][col] > 0:
            toe_loc_temp = col
    toe_loc += [float(toe_loc_temp)]

toe_location = pd.DataFrame( data={'toe_loc':toe_loc}, index=stream_hydrographs.index)
stream_toe_observed = pd.read_csv('./obs_data/stream_length.dat')
toe_location = toe_location.loc[stream_toe_observed[' Time [s]'].tolist(), :]

toe_location.to_csv('Simulated_streamfront.dat')