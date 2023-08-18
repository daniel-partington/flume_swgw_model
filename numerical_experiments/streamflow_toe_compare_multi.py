import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

flume_length = 8
horizontal_cells = 80

stream_toe_observed = pd.read_csv('./obs_data/stream_length.dat')
stream_toe_observed.loc[:, 'Flow [m$^3$/s]'] = stream_toe_observed['Flow [L/min]'] / 60. / 1000. 

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

folders = ['K_3.03E-4', 'K_3.73E-4']

stream_hydrographs = {}
toe_location = {}
for folder in folders:
    stream_hydrographs[folder] = pd.DataFrame()
    for index, x in enumerate(np.linspace(0.1, flume_length, horizontal_cells)):
        stream_hydrographs[folder].loc[:, str(x)] = get_hydrograph_data(os.path.join(folder, 'flumeo.hydrograph._x{}_.dat'.format(x)))['Surface']            
        
    toe_loc = []                           
    for row in stream_hydrographs[folder].iterrows():
        toe_loc_temp = 0.0
        for col in row[1].keys():
            if row[1][col] > 0:
                toe_loc_temp = col
        toe_loc += [float(toe_loc_temp)]
    
    toe_location[folder] = pd.DataFrame( data={'toe_loc':toe_loc}, index=stream_hydrographs[folder].index)

#def get_wb_data(fname):
#    wb = pd.read_csv(fname, skiprows=[0, 1, 2], delim_whitespace=True,
#                header=None, index_col=0, names=get_header(fname))
#    return wb[wb.columns[0:-1]]
#    
#
#wb = get_wb_data('flumeo.water_balance.dat')
#
#other_cols = ['Error rel', 'Error percent']
#plot_cols = [u'Fnodal_1', u'crit_depth', u'Helev_3', u'NET1 Sources/Sinks', u'PM',
#       u'Overland', u'NET2 Accumulation', u'ERROR (NET1-NET2)', u'Infilt', u'Exfilt']
#
##wb[plot_cols].plot()
#
#outflow = pd.read_excel('obs_data/Outflow_levels_march.xlsx')
#outflow = outflow[["Time (from internet, should match Calvin's computer)",
#                   'Qout m3/s']]
#outflow.columns = ['Time', 'Outflow [m$^3$/s]']                  
#today = datetime.date.today()
#combine = lambda x: datetime.datetime.combine(today, x)
#outflow.loc[:, 'ElapsedTime'] = outflow['Time'].apply(lambda x: combine(x) - combine(outflow['Time'].iloc[0]))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#---- PLOTTING ----------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from matplotlib.ticker import FormatStrFormatter

fig = plt.figure()
#ax = fig.add_subplot(2, 1, 1)
#wb.loc[:, 'Helev_3_neg'] = wb['Helev_3'] * -1
#wb[['Fnodal_1', 'Infilt', 'Helev_3_neg']].plot(ax=ax)
#ax.set_ylabel('Flux [m$^3$/s]', fontsize=10)
##ax.ticklabel_format(axis='y', style='sci')
##ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
#ax.set_xlabel('')
#ax.set_xticklabels('')
#ax.legend(['Inflow', 'Infiltration', 'Deep infiltration'], loc='upper left',
#          frameon=False, fontsize=10)
#ax.tick_params(axis='both', which='major', labelsize=10)
#ax.tick_params(axis='both', which='minor', labelsize=8)
#
ax = fig.add_subplot(1, 1, 1)
#ax.plot(toe_location[folders[0]].index.tolist(), 
#        toe_location[folders[0]]['toe_loc'].tolist(), 
#        color='grey')
#
#ax.plot(toe_location[folders[1]].index.tolist(), 
#        toe_location[folders[1]]['toe_loc'].tolist(), color='grey')
for folder in folders:
    toe_location[folder].plot(ax=ax)


ax.scatter(x=stream_toe_observed[' Time [s]'], y=stream_toe_observed[' Streamflow Length [m]'], 
           edgecolor='white', facecolor='red', s=40)
ax.legend([x.replace('_','=') for x in folders] + ['Observed'], loc='upper left', scatterpoints=1, 
          frameon=False, fontsize=10)
ax.set_ylabel('Length of stream [m]', fontsize=10)
ax.set_xlabel('Time [s]', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.set_ylim(0, 6)
stream_toe_observed[[' Streamflow Length [m]', ' Time [s]']].to_csv('Obs_toe_location.dat')
for folder in folders:
    toe_location[folder].to_csv('Sim_toe_location_{}.dat'.format(folder))

plt.savefig('obs_v_multi_k_sim_experiment2.png', dpi=300)