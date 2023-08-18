import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def get_hydrographs(folders):
    stream_hydrographs = {}
    for folder in folders:
        stream_hydrographs[folder] = pd.DataFrame()
        for index, x in enumerate(np.linspace(0.1, flume_length, horizontal_cells)):
            stream_hydrographs[folder].loc[:, str(x)] = get_hydrograph_data(os.path.join(folder, 'flumeo.hydrograph._x{}_.dat'.format(x)))['Surface']            
    return stream_hydrographs
    
def get_toe_locations(stream_hydrographs):            
    toe_location = {}
    for folder in stream_hydrographs:
        toe_loc = []                           
        for row in stream_hydrographs[folder].iterrows():
            toe_loc_temp = 0.0
            for col in row[1].keys():
                if row[1][col] > 0:
                    toe_loc_temp = col
            toe_loc += [float(toe_loc_temp)]
        
        toe_location[folder] = pd.DataFrame( data={'toe_loc':toe_loc}, index=stream_hydrographs[folder].index)
    return toe_location

def get_wb_data(fname):
    wb = pd.read_csv(fname, skiprows=[0, 1, 2], delim_whitespace=True,
                header=None, index_col=0, names=get_header(fname))
    return wb[wb.columns[0:-1]]
    
def get_water_balances(folders):
    wb = {}
    for folder in folders:
        wb[folder] = get_wb_data(os.path.join(folder, 'flumeo.water_balance.dat'))
    return wb
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#---- PLOTTING ----------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def plot_flow_v_toe(folders, toe_location, wb, locs, figsave_fname='flow_v_toe.png',
                    xlim=None, ylim=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for folder in folders:
        ax.plot(toe_location[folder].loc[locs, 'toe_loc'].tolist(), 
                wb[folder].loc[locs, "Fnodal_1"].tolist(), label=folder)#,
                #marker='o', markeredgecolor='none', linewidth=0)
    
    #ax.legend([x.replace('_',' ') for x in folders], loc='upper left', numpoints=1, 
    #          frameon=False, fontsize=10)
    ax.set_ylabel('Inflow rate [m$^3$/s]', fontsize=10)
    ax.set_xlabel('Length of stream [m]', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.savefig(figsave_fname, dpi=300)
    #plt.close()
    
if __name__ == '__main__':
    targets = ['n_x_10', 'n_x_0.1']
    locs = [1816., 3273., 4800., 7020., 9004., 10850., 12120., 13800.]
    stream_hydrographs = get_hydrographs(targets)
    toe_locations = get_toe_locations(stream_hydrographs)
    water_balances = get_water_balances(targets)
    toe_locations_max = max([toe_locations[key]['toe_loc'].max() for key in toe_locations])
    flow_max = max([water_balances[key]['Fnodal_1'].max() for key in water_balances])
    plot_flow_v_toe(targets, toe_locations, water_balances, locs, 
                    figsave_fname='flow_v_toe_K_val.png', 
                    xlim=[0, toe_locations_max], ylim=[0, flow_max])
    
    