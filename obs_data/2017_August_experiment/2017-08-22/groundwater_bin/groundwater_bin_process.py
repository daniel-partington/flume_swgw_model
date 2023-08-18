import pandas as pd
import numpy as np
import datetime

fname = "2017-08-22_2017-08-22_16-47-27-321.csv"
start = datetime.datetime(2017,8,22,12,34,4)

def get_header(fname):
    with open(fname, 'r') as f:
        header = f.readlines()[71].replace("   ", "").replace("  ", "").split(',')
        header = [h[:-1] if h[-1] == " " else h for h in header]
    return header
    
gw_bin_df = pd.read_csv(fname, parse_dates=[0], names=get_header(fname), skiprows=72).drop('\n', 1)
gw_bin_df = gw_bin_df[gw_bin_df['Date and Time'] >= start]
#gw_bin_df.plot(x='Seconds', y='Depth (m)')

# Smooth noisy data
gw_bin_df = gw_bin_df.resample('2T', on='Date and Time').mean()

dh = np.array([0] + gw_bin_df['Depth (m)'].tolist()) - np.array(gw_bin_df['Depth (m)'].tolist() + [0]) 
gw_bin_df.loc[:, 'dh'] = dh[1:]
gw_bin_df = gw_bin_df[gw_bin_df['dh'] < 0]
#gw_bin_df['dh'].plot()

def depth_to_volume(depth):
    # The grand clue to converting dpeth in bin to volume based on depth to water,
    # where the bin appears to be about 77.5 cm deep:
    #    y = -0.0026x + 0.1965
    #      
    volume = -0.0026 * (81.35 - depth * 100.) + 0.1965
    return volume

gw_bin_df.loc[:, 'volume'] = gw_bin_df.apply(lambda x: depth_to_volume(x['Depth (m)']), axis=1)
dV = np.array(gw_bin_df['volume'].tolist() + [0]) - np.array([0] + gw_bin_df['volume'].tolist()) 
gw_bin_df.loc[:, 'dV'] = dV[1:]

gw_bin_df['tvalue'] = gw_bin_df.index
gw_bin_df.loc[:, 'dt'] =  [t.total_seconds() for t in (gw_bin_df['tvalue'] - gw_bin_df['tvalue'].shift()).fillna(0)]

gw_bin_df.loc[:, 'Q'] = gw_bin_df['dV'] / gw_bin_df['dt']

#gw_bin_df = gw_bin_df[gw_bin_df['Q'] >= 0]

gw_bin_df[gw_bin_df.Q > 0]['Q'].plot(linewidth=0, marker='o', markeredgewidth=0.0)

gw_bin_df_filtered = gw_bin_df.copy()

gw_bin_df_filtered.replace(np.inf, np.nan, inplace=True)
gw_bin_df_filtered.dropna(inplace=True)

start_emptying = datetime.datetime(2017,8,22,13,15,4)
cutoff = datetime.datetime(2017,8,22,16,16,38)
y_cutoff = 1.1E-4

Q_std = gw_bin_df_filtered.describe()['Q'].loc['std']

import matplotlib.pyplot as plt

window = 5
std_mult = 1.2
df = gw_bin_df_filtered.copy()
df = df[df.Q > 0]
df['mean']= df['Q'].rolling(window).mean()
df['mean'] = df['mean'].bfill()
df['std'] = df['Q'].rolling(10).std()
df['std'] = df['std'].bfill()

#filter setup
#df_in = df[(df.Q <= df['mean'] + std_mult * df['std']) & (df.Q >= df['mean'] - std_mult * df['std'])]
#df_out = df[(df.Q > df['mean'] + std_mult * df['std']) | (df.Q < df['mean'] - std_mult * df['std'])]

df_in = df.copy()
df_in.loc[(df_in.index > start_emptying) & (df_in.index < cutoff)] = df_in.loc[(df_in.index > start_emptying) & (df_in.index < cutoff) & (df_in['Q'] > y_cutoff)]
df_in.dropna(inplace=True)
Q_grad = np.array(df_in['Q'].tolist() + [0]) - np.array([0] + df_in['Q'].tolist())
df_in.loc[:, 'Q_grad'] = Q_grad[:-1]                  
#df_in.plot(y=['Q', 'Q_grad'])
df_in.loc[df_in.index > start_emptying] = df_in.loc[(df_in.index > start_emptying) & (df_in.Q_grad > 0)]
df_in.dropna(inplace=True)
#df_in.plot(y='Q')                  
df_in = df_in[df_in.index < cutoff] 
                 
df_out = df.copy()             
             
fig = plt.figure()
ax = fig.add_subplot(111)
#upper = df['mean'] + std_mult * df['std']
#lower = df['mean'] - std_mult * df['std']
#ax.fill_between(df.index, upper, lower, color='grey', alpha=0.2)
df_in.Q.plot(ax=ax, linewidth=0, marker='o', markeredgewidth=0.0)
df_out.Q.plot(ax=ax, linewidth=0, marker='+', markeredgewidth=1.0)
ax.axvline(start_emptying, linestyle='--', alpha=0.5, color='grey')
ax.axvline(cutoff, linestyle='--', alpha=0.5, color='grey')
ax.axhline(y_cutoff, linestyle='--', alpha=0.5, color='grey')

df_in.loc[:,'time_elapsed'] = df_in.index - df_in.index[0]
df_in.loc[:,'time_elapsed'] = df_in.time_elapsed.apply(lambda x: x.seconds)
df_in.plot(x='time_elapsed', y='Q', linewidth=0, marker='o', markeredgewidth=0.0)

df_in[['time_elapsed', 'Q']].to_csv('groundwater_outflow.csv', index=False)
