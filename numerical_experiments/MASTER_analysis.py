import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import process_hydrographs_and_water_balances as phwb
import pandas as pd

'''

MASTER script to analyse all simulations for FLUME MODELLING project

Oberved Experiments:
    1. August 22nd 2017 - see Tom's thesis

Virtual Experiments:
    1. Multi-uniform K, n(porosity), alpha, beta, n(mannings)
    2. Multi-heterogeneous K with varying sigma
        Simple function to specify K(x) 
    3. Multi bed elevation with varying sigma
2
'''
verbose=True
load_pickle_only=True
results={}

theoretical_experiments = {'multi_uniform_var':True,
                           'multi_heterogeneous_k':True,
                           'multi_bed_elevation':True,
                           'multi_heterogeneous_mannings_n':False # Not easily implemented yet due to HGS limitations
                           }

master_analysis = {key:{} for key in theoretical_experiments if theoretical_experiments[key]}                           
                           
master_dir = os.getcwd()
samples = 30
flows = np.linspace(0, 3E-4, 31)
period_between_flow_changes = 2500.0
locs = np.arange(2500.0, 2500.0 * 31, 2500.0)

def mpd_to_mps(val):
    'Convert m/d to m/s'
    return val / 86400.            

uni_vars = {            
    # Soil properties (k~3.03E-4 m/s)
    'k': np.linspace(mpd_to_mps(1.), mpd_to_mps(50.), samples),  
    'porosity': np.linspace(0.2, 0.5, samples),
    'vg_alpha': np.linspace(0.8, 14.5, samples),
    'vg_beta': np.linspace(1.09, 2.68, samples),
    'vg_swr': np.linspace(0.045, 0.07, samples),
    # Bed properties
    'mannings_n': np.linspace(0.002, 0.2, samples)
    }
                            
hetero_vars = {'k': {'mu': 3.03E-4, 'sigmas': [1E-4, 1E-5, 1E-6]},
               'bed_elevation': {'mu': 0 , 'sigmas': [0.01, 0.001, 0.0001]},
               'mannings_n': {'mu': 0.024186654284876676, 'sigmas': [0.0001, 0.001, 0.01]}}

uni_vars_type = {
    # Soil properties
    'k': 'subsurface',  
    'porosity': 'subsurface',
    'vg_alpha': 'subsurface',
    'vg_beta': 'subsurface',
    'vg_swr': 'subsurface',
    # Bed properties
    'mannings_n': 'surface'
    }

requisite_files = ['array_sizes.default', 'batch.pfx', 'debug.control', 'flume.grok',
              'parallelindx.dat',
              'input_data/inflow.dat',
              'obs_locations/output_times.dat', 'obs_locations/outputs.dat',
              'properties/flume_riverbed.oprops', 'properties/material_sand.mprops']
requisite_folders = ['properties', 'obs_locations', 'obs_data', 'input_data']

def model_success(target):
    fname = os.path.join(target, 'flumeo.lst')
    if not os.path.exists(fname):
        return False
        
    with open(fname, 'r') as f:
        if '---- NORMAL EXIT ----' not in f.read():
            return False
        else:
            return True
            
def save_obj(filename, obj):
    with open(filename + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            
def load_obj(filename):
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            print "Loading: ", filename
            p = pickle.load(f)
            return p
        # End with
    else:
        raise TypeError('File type not recognised as "pkl": {}'.format(filename))
    # End if                

def process_outputs(targets, experiment, second_identifier):
    
    targets_good = []
    for target in targets:
        if model_success(target):
            targets_good += [target]
        else:
            print('Model failure for {}'.format(target))
            
    stream_hydrographs = phwb.get_hydrographs(targets_good)
    toe_locations = phwb.get_toe_locations(stream_hydrographs)
    water_balances = phwb.get_water_balances(targets_good)
    toe_locations_max = max([toe_locations[key]['toe_loc'].max() for key in toe_locations])
    flow_max = max([water_balances[key]['Fnodal_1'].max() for key in water_balances])
    save_obj('flow_v_toe_{}_{}'.format(experiment, second_identifier), [toe_locations, water_balances, targets_good])
    phwb.plot_flow_v_toe(targets_good, toe_locations, water_balances, locs, 
                         figsave_fname='flow_v_toe_{}_{}.png'.format(experiment, second_identifier), 
                         xlim=[0, toe_locations_max], ylim=[0, flow_max])              

if verbose:
    print('Analysing virtual experiments:')
for experiment in theoretical_experiments:
    if theoretical_experiments[experiment]:
        if verbose:
            print('    Analysing virtual experiment: {}'.format(experiment))
        if experiment == 'multi_uniform_var':
            for uni_var in uni_vars: 
                targets = []
                if verbose:
                    print('      Analysing runs for variable: {}'.format(uni_var))
                for val in uni_vars[uni_var]:
                    if verbose:
                        print('        Analysing run for variable val: {}'.format(val))
                    targets += ["{}_{}_{}".format(experiment, uni_var, val)]
                # end for
                if load_pickle_only:
                    results['{}_{}'.format(experiment, uni_var)] = load_obj('flow_v_toe_{}_{}.pkl'.format(experiment, uni_var))
                else:
                    process_outputs(targets, experiment, uni_var)
        elif experiment == 'multi_heterogeneous_k':
            for sigma in hetero_vars['k']['sigmas']:
                targets = []
                if verbose:
                    print('      Analysing sigma: {}'.format(sigma))
                for realisation in range(samples):
                    if verbose:
                        print('        Analysing realisation: {}'.format(realisation))
                    targets += ["{}_{}_{}".format(experiment, sigma, realisation)]
                if load_pickle_only:
                    results['{}_{}'.format(experiment, sigma)] = load_obj('flow_v_toe_{}_{}.pkl'.format(experiment, sigma))
                else:
                    process_outputs(targets, experiment, sigma)
        elif experiment == 'multi_heterogeneous_mannings_n':
            for sigma in hetero_vars['mannings_n']['sigmas']:
                for realisation in range(samples):
                    target = "{}_{}_{}".format(experiment, sigma, realisation)
        elif experiment == 'multi_bed_elevation':
            for sigma in hetero_vars['bed_elevation']['sigmas']:
                targets = []
                if verbose:
                    print('      Analysing sigma: {}'.format(sigma))
                for realisation in range(samples):
                    if verbose:
                        print('        Analysing realisation: {}'.format(realisation))
                    targets += ["{}_{}_{}".format(experiment, sigma, realisation)]
                if load_pickle_only:
                    results['{}_{}'.format(experiment, sigma)] = load_obj('flow_v_toe_{}_{}.pkl'.format(experiment, sigma))
                else:
                    process_outputs(targets, experiment, sigma)
    else:
        print("NOT executing: {}".format(experiment))
    # end if    
   
def discrete_cmap(N, base_cmap=None, limits=[0, 1]):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(limits[0], limits[1], N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

size_human_units = [115,85]
def mm_to_inches(num):
    if type(num) in [float, int]:
        return num * 0.0393701
    elif type(num) == list:
        return [x * 0.0393701 for x in num]
        
fig = plt.figure(figsize=mm_to_inches(size_human_units))
ax = fig.add_subplot(111)
toe_locations = results['multi_uniform_var_k'][0]
water_balances = results['multi_uniform_var_k'][1]
toe_locations_max = max([toe_locations[key]['toe_loc'].max() for key in toe_locations])
flow_max = max([water_balances[key]['Fnodal_1'].max() for key in water_balances])

k_values_map = {}
k_values = []
for folder in results['multi_uniform_var_k'][2]:
    k_values_map[folder] = float(folder.split('_')[-1])
    k_values += [float(folder.split('_')[-1])] 
min_var_val = min(k_values)
max_var_val = max(k_values)

norm = plt.Normalize(min_var_val, max_var_val)

for folder in results['multi_uniform_var_k'][2]:
    ax.plot(toe_locations[folder].loc[locs, 'toe_loc'].tolist(), 
            water_balances[folder].loc[locs, "Fnodal_1"].tolist(), c='grey', alpha=0.5)

for folder in results['multi_uniform_var_k'][2]:
    tmp = ax.scatter(x=toe_locations[folder].loc[locs, 'toe_loc'].tolist(), 
            y=water_balances[folder].loc[locs, "Fnodal_1"].tolist(), c=[k_values_map[folder]]*len(k_values), label=folder,
            cmap='viridis', norm=norm, s=50, edgecolor='none', alpha=1.0, vmin=min_var_val, vmax=max_var_val)#,
            #marker='o', markeredgecolor='none', markerfacecolor='blue', linewidth=0)
#ax.legend([x.replace('_',' ') for x in folders], loc='upper left', numpoints=1, 
#          frameon=False, fontsize=10)

ax.set_ylabel('Inflow rate [10$^{-4}$ m$^3$/s]', fontsize=10)
ax.set_xlabel('Length of stream [m]', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.set_xlim(0, toe_locations_max)
ax.set_ylim(0, flow_max)

cbar = plt.colorbar(tmp)
cbar.ax.tick_params(labelsize=8)
cbar_labels = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels([str(float(yi.get_text()) * 10000) for yi in cbar_labels if yi.get_text() != ''])
cbar.set_label('K [10$^{-4}$ m/s]', rotation=270, fontsize=10, labelpad=13)

plt.subplots_adjust(top=0.98, bottom=0.13, left=0.14, right=0.98)

plt.savefig('uniform_K_sampling.png', dpi=300)    

new_yticklabels = [str(float(yi.get_text()) * 10000) for yi in ax.get_yticklabels() if yi.get_text() != '']
ax.set_yticklabels(new_yticklabels)

plt.savefig('uniform_K_sampling.png', dpi=300)    

#### K heterogeneous

fig = plt.figure(figsize=mm_to_inches(size_human_units))
ax = fig.add_subplot(111)
three_col = ['r', 'g', 'b']
k_sigma_results = {}
for num, k_sigma in enumerate(hetero_vars['k']['sigmas']):
    k_sigma_results[k_sigma] = results['multi_heterogeneous_k_{}'.format(k_sigma)]
    toe_loc_tmp = results['multi_heterogeneous_k_{}'.format(k_sigma)][0]
    water_bal_tmp = results['multi_heterogeneous_k_{}'.format(k_sigma)][1]
    toe_loc_max = 0
    flow_max = 0
    for ind, mkey in enumerate(toe_loc_tmp):
        toe_loc_max_tmp = max([toe_loc_tmp[key]['toe_loc'].max() for key in toe_loc_tmp])
        flow_max_tmp = max([water_bal_tmp[key]['Fnodal_1'].max() for key in water_bal_tmp])
        if toe_loc_max_tmp > toe_loc_max:
            toe_loc_max = toe_loc_max_tmp
        if flow_max_tmp > flow_max:
            flow_max = flow_max_tmp
        tl = toe_loc_tmp[mkey].loc[locs, 'toe_loc'].tolist() 
        wb = water_bal_tmp[mkey].loc[locs, "Fnodal_1"].tolist()
        if ind == 0:
            combined_toes = pd.DataFrame(data={mkey.split('_')[-1]:tl}, index=wb)
        else:
            combined_toes.loc[:, mkey.split('_')[-1]] = tl
    upper = combined_toes.max(axis=1).tolist()
    lower = combined_toes.min(axis=1).tolist()
    ax.fill_betweenx(wb, lower, upper, alpha=0.5, label='{0: 1.1e}'.format(float(k_sigma)), color=three_col[num])
ax.legend(fontsize=10, loc='upper left', frameon=False, title= '$\sigma _{K}$')    
ax.set_ylabel('Inflow rate [10$^{-4}$ m$^3$/s]', fontsize=10)
ax.set_xlabel('Length of stream [m]', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.set_xlim(0, toe_loc_max)
ax.set_ylim(0, flow_max)

plt.subplots_adjust(top=0.98, bottom=0.13, left=0.14, right=0.98)

left, bottom, width, height = [0.62, 0.24, 0.33, 0.30]
ax2 = fig.add_axes([left, bottom, width, height])
for num, k_sigma in enumerate(hetero_vars['k']['sigmas']):
    ax2.hist(np.random.normal(hetero_vars['k']['mu'], k_sigma, samples * 161), bins=80, color=three_col[num], alpha=0.5, linewidth=0)

ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_xlabel('K [10$^{-4}$ m/s]', fontsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.tick_params(axis='both', which='minor', labelsize=8)
ax2.set_xticks(np.arange(0.0001, 0.0006, step=0.0002))
#ax2_lim = ax2.get_xlim()
ax2.set_xlim(0, 0.0006)#ax2_lim[1])
ax2.set_ylim(0, 200)
ax2.xaxis.labelpad = -2
ax2.yaxis.labelpad = 2
    
plt.savefig('heterogeneous_K_sampling.png', dpi=300)    

new_yticklabels = [str(float(yi.get_text()) * 10000) for yi in ax.get_yticklabels() if yi.get_text() != '']
ax.set_yticklabels(new_yticklabels)

ax2_xlabels = ax2.get_xticklabels()
ax2.set_xticklabels([str(float(xi.get_text()) * 10000) for xi in ax2_xlabels if xi.get_text() not in ['',  u'\u2212']])
ax2.set_yticklabels('')    

plt.savefig('heterogeneous_K_sampling.png', dpi=300)     

#### Bed elevation heterogeneous

fig = plt.figure(figsize=mm_to_inches(size_human_units))
ax = fig.add_subplot(111)
three_col = ['r', 'g', 'b']
be_sigma_results = {}
for num, be_sigma in enumerate(hetero_vars['bed_elevation']['sigmas']):
    be_sigma_results[be_sigma] = results['multi_bed_elevation_{}'.format(be_sigma)]
    toe_loc_tmp = results['multi_bed_elevation_{}'.format(be_sigma)][0]
    water_bal_tmp = results['multi_bed_elevation_{}'.format(be_sigma)][1]
    toe_loc_max = 0
    flow_max = 0
    for ind, mkey in enumerate(toe_loc_tmp):
        toe_loc_max_tmp = max([toe_loc_tmp[key]['toe_loc'].max() for key in toe_loc_tmp])
        flow_max_tmp = max([water_bal_tmp[key]['Fnodal_1'].max() for key in water_bal_tmp])
        if toe_loc_max_tmp > toe_loc_max:
            toe_loc_max = toe_loc_max_tmp
        if flow_max_tmp > flow_max:
            flow_max = flow_max_tmp
        tl = toe_loc_tmp[mkey].loc[locs, 'toe_loc'].tolist() 
        wb = water_bal_tmp[mkey].loc[locs, "Fnodal_1"].tolist()
        if ind == 0:
            combined_toes = pd.DataFrame(data={mkey.split('_')[-1]:tl}, index=wb)
        else:
            combined_toes.loc[:, mkey.split('_')[-1]] = tl
    upper = combined_toes.max(axis=1).tolist()
    lower = combined_toes.min(axis=1).tolist()
    ax.fill_betweenx(wb, lower, upper, alpha=0.5, label='{0: 1.1e}'.format(float(be_sigma)), color=three_col[num])
ax.legend(fontsize=10, loc='upper left', frameon=False, title= '$\sigma _{bed\ elevation}$')    
ax.set_ylabel('Inflow rate [10$^{-4}$ m$^3$/s]', fontsize=10)
ax.set_xlabel('Length of stream [m]', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.set_xlim(0, toe_loc_max)
ax.set_ylim(0, flow_max)

plt.subplots_adjust(top=0.98, bottom=0.13, left=0.14, right=0.98)

left, bottom, width, height = [0.62, 0.24, 0.33, 0.30]
ax2 = fig.add_axes([left, bottom, width, height])
for num, be_sigma in enumerate(hetero_vars['bed_elevation']['sigmas']):
    ax2.hist(np.random.normal(hetero_vars['bed_elevation']['mu'], be_sigma, samples * 161), bins=80, color=three_col[num], alpha=0.5, linewidth=0)

ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_xlabel('Deviation [10$^{-2}$ m]', fontsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.tick_params(axis='both', which='minor', labelsize=8)
ax2.set_xticks(np.arange(-0.03, 0.04, step=0.02))
#ax2_lim = ax2.get_xlim()
ax2.set_xlim(-0.03, 0.03)#ax2_lim[1])
ax2.set_ylim(0, 200)
ax2.xaxis.labelpad = -2
ax2.yaxis.labelpad = 2

plt.savefig('heterogeneous_bed_elevation_sampling.png', dpi=300)    

new_yticklabels = [str(float(yi.get_text()) * 10000) for yi in ax.get_yticklabels() if yi.get_text() != '']
ax.set_yticklabels(new_yticklabels)

ax2_xlabels = ax2.get_xticklabels()
ax2.set_xticklabels([str(float(xi.get_text().replace(u'\u2212', '-')) * 100) for xi in ax2_xlabels if xi.get_text() not in ['',  u'\u2212']])
ax2.set_yticklabels('')    

plt.savefig('heterogeneous_bed_elevation_sampling.png', dpi=300) 