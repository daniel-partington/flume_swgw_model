import os
from shutil import copy2
import subprocess

import numpy as np

import grok_gen as gg
import grok_helper_functions as ghf

np.random.seed(1922)

'''

MASTER script to run and launch all simulations for FLUME MODELLING project

Oberved Experiments:
    1. August 22nd 2017 - see Tom's thesis

Virtual Experiments:
    1. Multi-uniform K, n(porosity), alpha, beta, n(mannings)
    2. Multi-heterogeneous K with varying sigma
        Simple function to specify K(x) 
    3. Multi bed elevation with varying sigma

'''
verbose=True

theoretical_experiments = {'multi_uniform_var':True,
                           'multi_heterogeneous_k':False,
                           'multi_bed_elevation':False,
                           'multi_heterogeneous_mannings_n':False # Not easily implemented yet due to HGS limitations
                           }

master_dir = os.getcwd()
slope = -0.003
samples = 30
flows = np.linspace(0, 3E-4, 31)
period_between_flow_changes = 2500.0

ghf.gen_input_for_inflows('input_data/inflow.dat', flows, period_between_flow_changes)
ghf.gen_output_times('obs_locations/output_times.dat', flows, period_between_flow_changes)

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
               'bed_elevation': {'mu': 0 , 'sigmas': [0.0001, 0.001, 0.01]},
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
              
def write_slurm_job_script(fname, master_dir, job_name, ntasks=1, mem=8, time=24):
    with open(fname, 'w') as f:
        f.write('''#!/bin/bash

### Job Name
#SBATCH --job-name={0}

### Set email type for job
### Accepted options: NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-type=NONE

### email address for user
#SBATCH --mail-user=part0075@flinders.edu.au
### Queue name that job is submitted to
#SBATCH --partition=tango

### Request nodes
#SBATCH --ntasks={1}
#SBATCH -c 1
#SBATCH --mem={2}gb
#SBATCH --time={3}:00:00

rm -Rf /scratch/danpartington/{0} 
mkdir /scratch/danpartington/{0}
cp -Rf {4}/{0} /scratch/danpartington
cd /scratch/danpartington/{0}

grok.x

wait

phgs.x

wait

tail -n 57 flumeo.list > lst_last.txt 
cp -Rf /scratch/danpartington/{0} {4} 
rm -rf /scratch/danpartington/{0} 
    
        '''.format(job_name, ntasks, mem, time, master_dir))              

def gen_folders_and_copy_files(target):
    try:
        os.mkdir(target)
    except:
        print('Could not create {}'.format(target))
    for requisite_folder in requisite_folders:
        try:
            os.mkdir(os.path.join(target, requisite_folder))
        except:
            print('Could not create {}'.format(requisite_folder))
    for requisite_file in requisite_files:
        if '/' in requisite_file:
            copy2(requisite_file, os.path.join(target,requisite_file.split('/')[0]))
        else:
            copy2(requisite_file, target)
    waiting = True
    while waiting:
        if os.path.exists(target):
            os.chdir(target)
            waiting=False
            

def write_and_launch_job(master_dir, target, verbose=True):
    #os.chdir(target)
    write_slurm_job_script('job.sub', master_dir, target)
    waiting = True
    while waiting:
        if os.path.exists('job.sub'):
            waiting=False
    if verbose:
        print('        Launching from: {}'.format(os.getcwd()))
    subprocess.call('sbatch job.sub', shell=True)
    os.chdir(master_dir)

if verbose:
    print('Launching virtual experiments:')
for experiment in theoretical_experiments:
    if theoretical_experiments[experiment]:
        if verbose:
            print('    Launching virtual experiment: {}'.format(experiment))
        print("Executing: {}".format(experiment))
        if experiment == 'multi_uniform_var':
            for uni_var in uni_vars: 
                if verbose:
                    print('      Launching variable: {}'.format(uni_var))
                for val in uni_vars[uni_var]:
                    if verbose:
                        print('        Launching variable val: {}'.format(val))
                    target = "{}_{}_{}".format(experiment, uni_var, val)
                    mprop_fname = os.path.join(master_dir, target, 'properties', 'material_sand.mprops')
                    oprop_fname = os.path.join(master_dir, target, 'properties', 'flume_riverbed.oprops')
                    gen_folders_and_copy_files(target)
                    if uni_vars_type[uni_var] == 'subsurface':
                        if uni_var == 'k':
                            ghf.gen_mprops(k=val,
                                           fname=mprop_fname)
                        if uni_var == 'porosity':
                            ghf.gen_mprops(porosity=val,
                                           fname=mprop_fname)
                        if uni_var == 'vg_alpha':
                            ghf.gen_mprops(alpha=val,
                                           fname=mprop_fname)
                        if uni_var == 'vg_beta':
                            ghf.gen_mprops(beta=val,
                                           fname=mprop_fname)
                        if uni_var == 'vg_swr':
                            ghf.gen_mprops(residual_saturation=val,
                                           fname=mprop_fname)
                        ghf.gen_oprops(fname=oprop_fname)
                    elif uni_vars_type[uni_var] == 'surface':
                        if uni_var == 'mannings_n':
                            ghf.gen_oprops(mannings_n=val,
                                           fname=oprop_fname)
                        ghf.gen_mprops(fname=mprop_fname)
                    gg.generate_grok_for_flume_model(
                                  'flume.grok',
                                  inflow_data='inflow.dat',
                                  heterogeneous_k=False)
                    write_and_launch_job(master_dir, target)
                # end for
        elif experiment == 'multi_heterogeneous_k':
            for sigma in hetero_vars['k']['sigmas']:
                if verbose:
                    print('      Launching sigma: {}'.format(sigma))
                for realisation in range(samples):
                    if verbose:
                        print('        Launching realisation: {}'.format(realisation))
                    target = "{}_{}_{}".format(experiment, sigma, realisation)
                    gen_folders_and_copy_files(target)
                    x = np.array([hetero_vars['k']['mu']] * 161)
                    ghf.gen_raster_top(x, 0, sigma, 'spatial_k.dat')
                    gg.generate_grok_for_flume_model(
                                  'flume.grok',
                                  inflow_data='inflow.dat',
                                  heterogeneous_k=True)
                    write_and_launch_job(master_dir, target)
        elif experiment == 'multi_heterogeneous_mannings_n':
            for sigma in hetero_vars['mannings_n']['sigmas']:
                for realisation in range(samples):
                    target = "{}_{}_{}".format(experiment, sigma, realisation)
                    gen_folders_and_copy_files(target)
                    x = np.array([hetero_vars['mannings_n']['mu']] * 161)
                    ghf.gen_raster_top(x, 0, sigma, 'spatial_n.dat')
                    gg.generate_grok_for_flume_model(
                                  'flume.grok',
                                  inflow_data='inflow.dat',
                                  heterogeneous_n=True)
                    write_and_launch_job(master_dir, target)
        elif experiment == 'multi_bed_elevation':
            for sigma in hetero_vars['bed_elevation']['sigmas']:
                if verbose:
                    print('      Launching sigma: {}'.format(sigma))
                for realisation in range(samples):
                    if verbose:
                        print('        Launching realisation: {}'.format(realisation))
                    target = "{}_{}_{}".format(experiment, sigma, realisation)
                    gen_folders_and_copy_files(target)
                    x = np.array([0.45 + xi * slope for xi in np.linspace(-0.025, 8.025, 162)])
                    minx = np.array([0.44 + xi * slope for xi in np.linspace(-0.025, 8.025, 162)])
                    ghf.gen_raster_top(x, 0, sigma, 'riverbed_elevation.dat', minx=minx, pad=False)
                    gg.generate_grok_for_flume_model(
                                  'flume.grok',
                                  varying_surface=True,
                                  inflow_data='inflow.dat'
                                  )
                    write_and_launch_job(master_dir, target)

    else:
        print("NOT executing: {}".format(experiment))
    # end if    