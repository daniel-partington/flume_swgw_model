import os
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import datetime
import pandas as pd
import numpy as np
import interpolation

p_j = os.path.join
obs_folder = 'obs_data/2017 August experiment/2017-08-22/soil moisture'
file_format = 'dat'

'# Defining Sensor Locations'
x = [0.5, 1.5, 2.5, 3.5] * 2 + \
    [0.5, 1.5, 3.5] + \
    [0.5, 2.5, 3.5] * 2 + \
    [0.5, 1.5, 3.5] * 2 + \
    [0.5, 1.5]
y = [0.05] * 4 + [0.1] * 4 + [0.15] * 3 + [0.2] * 3 + [0.25] * 3 + [0.3] * 3 + \
    [0.35] * 3 + [0.4] * 2

'# Setting Experiment Start Time and Elapsed Time to be Examined'
elapsed_time = 13000
start = datetime.datetime(2017,8,22,12,34,4)


'# Importing Soil Moisture Sensor Data into Data Frames'
files = ['CR1000_36091_Table1.{}'.format(file_format), 
         'CR1000_36146_Table1.{}'.format(file_format), 
         'CR1000_80326_Table1.{}'.format(file_format), 
         'CR3000_flume_Table1.{}'.format(file_format), 
         'CR3000_FU_Table1.{}'.format(file_format)]

'# Logger Sampling Frequencies (sec)'
sampling_frequencies = {'CR1000_36091': 2,
                        'CR1000_36146': 5,
                        'CR1000_80326': 2,
                        'CR3000_flume': 5,
                        'CR3000_FU': 5}

names = [fle.split('.')[0].replace('_Table1', '') for fle in files]
         
df_data = {}       
sensors = {}
  
for fle in files:         
    name = fle.split('.')[0].replace('_Table1', '')
    if file_format in ['xlsx', 'xls']:
        df_data[name] = pd.read_excel(p_j(obs_folder, fle), header=1, na_values='NAN')
    elif file_format in ['dat', 'csv']:
        df_data[name] = pd.read_csv(p_j(obs_folder, fle), na_values='NAN', skiprows=[0, 2, 3], index_col='TIMESTAMP')

    # Some sensors have VWC and some have VW data ...
    # this next bit of code places preference on the VWC column if both exist
    # and doesn't rename the column with VW. This is important as only columns
    # beginning with and 'S' are processed further
    tmp_vwc = [c.replace("VWC_", "") for c in df_data[name].columns]
    tmp_vc = [c.replace("VW_", "").replace("_Avg", "") for c in df_data[name].columns]
    duplicates = list(set([s for s in tmp_vwc if s[0] == 'S']).intersection( 
                 [s for s in tmp_vc if s[0] == 'S']))
    # Deal with CR1000 naming of sensors
    df_data[name].columns = [c.replace("VWC_", "") for c in df_data[name].columns]
    # Deal with CR3000 naming of sensors
    tmp_col = [c.replace("VW_", "").replace("_Avg", "") if \
               c.replace("VW_", "").replace("_Avg", "") not in duplicates else \
               c for c in df_data[name].columns ]
    df_data[name].columns = tmp_col
    for col in df_data[name].columns:
        if col[0] == 'S':
            sensors[col] = {'name':name}
    df_data[name]['TIMESTAMP'] = df_data[name].index
    df_data[name]['TIMESTAMP'] = pd.to_datetime(df_data[name]['TIMESTAMP'])
    for col in df_data[name].columns:
        if col not in ['TIMESTAMP', 'RECORD']:
            df_data[name][col]= df_data[name][col].astype(float)

if not start:
    for index, name in enumerate(names):
        if index == 0:
            start = df_data[name].loc[0, 'TIMESTAMP']
        else:
            if df_data[name].loc[0, 'TIMESTAMP'] < start:
                start = df_data[name].loc[0, 'TIMESTAMP']
        # end if
    # end for
# end if

'# Appending an Elapsed Time Column to Data Frames'
logger_sensor_map = {}
sensor_logger_map = {}
for name in names:
    df_data[name].loc[:, 'Elapsed Time'] = [t.total_seconds() for t in (df_data[name]['TIMESTAMP'] - start)]
    logger_sensor_map[name] = [s for s in df_data[name].columns if s[0] == 'S']
    for s in df_data[name].columns:
        if s[0] == 'S':    
            sensor_logger_map[s] = name
# end  for

# Create logger sensor map dict


# Conversion to % saturation is done via an equation of the form:
# sat% = (x + a) / (b - c) * 100
saturation_conversion_functions = {
     'S1_8': lambda x: (x - 1.83) / (37.58 - 1.83) * 100,
     'S2_8': lambda x: (x - 2.26) / (41.71 - 2.26) * 100,
     'S3_8': lambda x: (x - 2.08) / (39.05 - 2.08) * 100,
     'S4_8': lambda x: (x - 0.002) / (0.112 - 0.002) *100,
     'S1_7': lambda x: (x - 4.91) / (37.75 - 4.91) * 100,
     'S2_7': lambda x: (x - 6.17) / (38.48 - 6.17) * 100,
     'S3_7': lambda x: (x - 6.53) / (37.58 - 6.53) * 100,
     'S4_7': lambda x: (x - 0.010) / (0.122 - 0.010) * 100,
     'S1_6': lambda x: (x - 6.47) / (37.47 - 6.47) * 100,
     'S2_6': lambda x: (x - 8.03) / (36.34 - 8.03) * 100,
     'S4_6': lambda x: (x - 0.016) / (0.117 - 0.016) * 100,
     'S1_5': lambda x: (x - 8.08) / (37.35 - 8.08) * 100,
     'S3_5': lambda x: (x - 0.014) / (0.134 - 0.014) * 100,
     'S4_5': lambda x: (x - 0.016) / (0.137 - 0.016) * 100,
     'S1_4': lambda x: (x - 9.82) / (38.60 - 9.82) * 100,
     'S3_4': lambda x: (x - 0.058) / (0.331 - 0.058) * 100,
     'S4_4': lambda x: (x - 0.016) / (0.133 - 0.016) * 100,
     'S1_3': lambda x: (x - 9.28) / (38.06 - 9.28) * 100,
     'S2_3': lambda x: (x - 9.40) / (37.70 - 9.40) * 100,
     'S4_3': lambda x: (x - 0.015) / (0.126 - 0.015) * 100,
     'S1_2': lambda x: (x - 8.98) / (37.35 - 8.98) * 100,
     'S2_2': lambda x: (x - 9.82) / (35.88 - 9.82) * 100,
     'S4_2': lambda x: (x - 0.017) / (0.123 - 0.017) * 100,
     'S1_1': lambda x: (x - 8.15) / (38.43 - 8.15) * 100,
     'S2_1': lambda x: (x - 9.82) / (39.11 - 9.82) * 100}

# Specify the sensors that worked to allow filtering of bad ones
sensors_of_interest = ['S1_8', 'S2_8', 'S3_8', 'S4_8', 
                       'S1_7', 'S2_7', 'S3_7', 'S4_7', 
                       'S1_6', 'S2_6', 'S4_6',
                       'S1_5', 'S3_5', 'S4_5',
                       'S1_4', 'S3_4', 'S4_4',
                       'S1_3', 'S2_3', 'S4_3',
                       'S1_2', 'S2_2', 'S4_2',
                       'S1_1', 'S2_1']

def sensor_vals_at_elapsed_time(elapsed_time, df_data, sensors, sampling_frequencies, 
                                sensors_of_interest, saturation_conversion_functions):
    '# Extracting Sensor Data at Elapsed Time'
    for name in names:
        df_data[name].loc[:, 'Elapsed Time diff'] = np.abs(df_data[name]['Elapsed Time'] - elapsed_time)           
        df = df_data[name]
        df.index = df['Elapsed Time']
        df[df['Elapsed Time diff'] <= sampling_frequencies[name] / 2.][[col for col in df.columns if col[0] =='S'] + ['Elapsed Time']]
        closest_indices = np.sort(df['Elapsed Time diff'].copy().sort_values().index.tolist()[0:2])
        if elapsed_time in closest_indices:
            closest_indices = [elapsed_time]
            for col in df_data[name].columns:
                if col[0] == 'S':
                    sensors[col][elapsed_time] = df.loc[elapsed_time, col]
        else:
            df_relevant = df.loc[closest_indices]
            df_relevant = df_relevant.append(pd.DataFrame({cod:np.nan for cod in df_relevant.columns}, index=[elapsed_time]))
            df_relevant = df_relevant.reindex(df_relevant.index.sort_values())
            df_relevant.interpolate(method='index', inplace=True)
            for col in df_data[name].columns:
                if col[0] == 'S':
                    sensors[col][elapsed_time] = df_relevant.loc[elapsed_time, col]
    
    # Saturation Percentage Vector       

    z = [saturation_conversion_functions[sensor](sensors[sensor][elapsed_time]) for sensor in sensors_of_interest]
 
    return sensors, z 
     
elapsed_times = [0, 100, 200, 300, 500, 700, 1000, 1200, 1500, 1800]
#for elapsed_time in elapsed_times:
#    sensors, z = sensor_vals_at_elapsed_time(elapsed_time, df_data, sensors, sampling_frequencies,
#                                             sensors_of_interest, saturation_conversion_functions)
#    # Plotting Soil Moisture Profiles
#    fig = plt.figure(figsize=(12, 5))
#    ax = fig.add_subplot(111)
#    soil_moisture = ax.scatter(x, y, c=z, s=100, vmin=0, edgecolor='none', vmax=100, cmap='viridis')
#    cbar = fig.colorbar(soil_moisture) #, shrink=0.2, aspect=5)
#    cbar.set_label('% Saturation', rotation=270)
#    plt.gca().invert_yaxis()
#    plt.title('Elapsed time: {} sec'.format(elapsed_time))
#    plt.xlabel('Horizontal Location (m)')
#    plt.ylabel('Depth (m)')
#    #plt.gca().set_aspect('equal', adjustable='box')
#    plt.yticks([0, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450])
#    #plt.savefig(p_j(obs_folder, 'TimeTest.png'), dpi=300)
#    plt.subplots_adjust(top=0.93, bottom=0.11, left=0.06, right=1)

# Read in the modelling results:
prefix = 'flume'
sim_fname_dict = {sensor:'{}o.observation_well_flow.{}.dat'.format(prefix, sensor.replace('_','-')) for sensor in sensors_of_interest} 
sim_dfs = {}

def get_header(fname):
    with open(fname, 'r') as f:
        header = f.readlines()[1].split('=')[1].replace('"', '').replace('\n', '').replace(' ', '').split(',')
    return header
    
for key in sim_fname_dict:
    sim_dfs[key] = pd.read_csv(sim_fname_dict[key], names=get_header(sim_fname_dict[key]), skiprows=2, delim_whitespace=True)
    
## Individual plots
#for sensor in sensors_of_interest:
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    sim = sim_dfs[sensor]
#    ax.plot(sim['Time'], sim['S'] * 100., label='Simulated')
#    obs = df_data[sensor_logger_map[sensor]][['Elapsed Time', sensor]]
#    obs = obs[obs['Elapsed Time'] >= 0.]
#    obs[sensor] = obs[sensor].apply(saturation_conversion_functions[sensor])
#    ax.plot(obs['Elapsed Time'], obs[sensor], label='Observed')
#    ax.set_ylim(0, 110)
#    ax.legend()
#    ax.set_title(sensor)

max_c = max([s[1] for s in sensors_of_interest])
max_r = max([s[3] for s in sensors_of_interest])    

# Combined plot
fig = plt.figure(figsize=(12, 10))
for index, sensor in enumerate(sensors_of_interest):
    ax = fig.add_subplot(max_r, max_c, int(sensor[1]) +  int(max_c) * (int(sensor[3]) - 1))
    sim = sim_dfs[sensor]
    sim.loc[:,'Time [hr]'] = sim['Time'] / 60.0 / 60.
    ax.plot(sim['Time [hr]'], sim['S'] * 100., label='Simulated')
    obs = df_data[sensor_logger_map[sensor]][['Elapsed Time', sensor]]
    obs = obs[obs['Elapsed Time'] >= 0.]
    obs.loc[:,'Time [hr]'] = obs['Elapsed Time'] / 60.0  / 60.
    obs[sensor] = obs[sensor].apply(saturation_conversion_functions[sensor])
    ax.plot(obs['Time [hr]'], obs[sensor], label='Observed')
    ax.set_ylim(0, 110)
    if int(sensor[1]) > 1:
        ax.set_yticklabels('')
    elif int(sensor[3]) == 4:
        ax.set_ylabel('% Saturation')
    if int(sensor[3]) < int(max_r):
        ax.set_xticklabels('')
    elif int(sensor[1]) == 2:
        ax.set_xlabel('Elasped time [hr]')
    #ax.legend()
    ax.set_title(sensor, fontsize=10)
fig.subplots_adjust()    

# Setup the initial saturation for the model
sensors, z = sensor_vals_at_elapsed_time(0, df_data, sensors, sampling_frequencies,
                                         sensors_of_interest, saturation_conversion_functions)

y_elev = np.array([0.45 - xi * 0.003 for xi in x]) - np.array(y) 

#plt.scatter(x, y_elev, c=z)

x_model_resolution = 8.0 / 160
y_model_resolution = 0.45 / 45


grid_x, grid_y = np.mgrid[0:8:161j, 0:0.45:46j]
xi = (grid_x, grid_y)
points = np.array(zip(x, y_elev))

initial_sat_2D = interpolation.Interpolator('structured', points, z, xi, method='linear')
initial_sat_2D[initial_sat_2D > 100.] = 100.
initial_sat_2D[initial_sat_2D < 0.] = 0.

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

fig = plt.figure(figsize=(8,3))
ax = fig.add_subplot(111)
gp = ax.imshow(initial_sat_2D.T, extent=(0, 8, 0, 0.45), origin='lower', 
               cmap='viridis', interpolation='none', aspect='auto',
               vmin=0, vmax=100, clip_on=True, alpha=0.6)
x1, x2, y1, y2 = gp.get_extent()
gp._image_skew_coordinate = (x2, y1)
x1, x2, y1, y2 = gp.get_extent()
gp._image_skew_coordinate = (x2, y1)
trans_data = mtransforms.Affine2D().skew(0, -np.tan(3.0/1000.0)) + ax.transData
gp.set_transform(trans_data)

cbar = plt.colorbar(gp, orientation='vertical', fraction=.1)
cbar.ax.set_ylabel('Saturation (%)')

x1, x2, y1, y2 = gp.get_extent()
#ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--", transform=trans_data)
ax.set_xlim(0, 8)
ax.set_ylim(-0.1, 0.5)

ax.plot(x, y_elev, linewidth=0, marker='o', markerfacecolor='white', markeredgecolor='black')

ax.axes.tick_params(direction='out')

plt.savefig('initial_saturation.png', dpi=300)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#plt.title('Initial saturation')

initial_sat_3D = np.stack([initial_sat_2D, initial_sat_2D])
initial_sat_3D.shape
plt.imshow(initial_sat_3D[0][:][:].T, extent=(0, 8, 0, 0.45))

initial_sat_3D_reshape = initial_sat_3D.reshape(161, 2, 46)
initial_sat_3D_reshape.shape
plt.figure()
plt.imshow(initial_sat_3D[:][0][:].T, extent=(0, 8, 0, 0.45))

#def saturation_to_head()
initial_sat_3D_reshape[0][0][0]
initial_sat_3D_reshape[0][1][0]
initial_sat_3D_reshape[160][0][44]
initial_sat_3D_reshape[160][1][44]

x_node = np.linspace(0, 8, 161)
y_node = np.linspace(0, 0.45, 46)
z_node = []

for i in x_node:
    for j in y_node:
        z_node = []

#y_elev = np.array([0.45 - xi * 0.003 for xi in x_all]) - np.array(y) 

def van_genuchten_sat_from_pressure(pressure, swr=5E-2, alpha=12.0539, beta=5.67, gamma=0.824, 
                                    min_p=-1000):
    if pressure < 0:
        sw = swr + (1 - swr) * (1 + np.abs(alpha * pressure) ** beta) ** (-(1 - 1/beta))
    else:
        sw = 1
    # end if
    return sw

def van_genuchten_pressure_from_sat(sw, swr=5E-2, alpha=12.0539, beta=5.67, 
                                    gamma=0.824, min_p=-1000):
    pressure_ = np.power((sw - swr) / (1. - swr), (1. - 1. / beta))
    pressure_ = pressure_ - 1.
    print pressure_
    return pressure_
    new_beta = -1 * beta
    pressure = np.power(pressure_, new_beta)
    print pressure
    #/ abs(alpha) 
    #if pressure < min_p:
    #    pressure = min_p
    # end if
    #return pressure

def pressure_from_saturation(sw, swr=5E-2, alpha=12.0539, beta=5.67, gamma=0.824):
    '''
    !
    !  ...Compute the pressure for water saturation, sw
    !     Written by Kerry MacQuarrie, Nov. 1995
    !
    '''

    c1 = 1.0E0
    eps_conv = 1.0E-4
    sat_shift = 1.0E-6
    aconstant = c1 - eps_conv

    #! Compute effective saturation
    se = (sw - swr) / (c1 - swr)
    # Compute some constants
    v_gamma = c1 / gamma
    v_beta  = c1 / beta
    v_alpha = c1 / alpha

    if (se > aconstant): # use linear interpolation
        sat_1  = aconstant
        pres_1 = v_alpha * ( (sat_1) ** (-v_gamma) - c1 ) ** (v_beta)
        sat_2  = c1 # note that the pressure at sat=1.0 is 0.0
        dp_ds  = pres_1 / eps_conv
        pw_from_sw = dp_ds * (se - sat_1) + pres_1

    elif se < eps_conv: # use linear interpolation

        sat_1  = eps_conv
        pres_1 = v_alpha * ( (sat_1) ** (-v_gamma) - c1 ) ** (v_beta)
        sat_2  = eps_conv + sat_shift
        pres_2 = v_alpha * ( (sat_2)**(-v_gamma) - c1 ) ** (v_beta)
        dp_ds  = (pres_2 - pres_1) / sat_shift
        pw_from_sw = dp_ds * (se - sat_1) + pres_1

    else: # invert van Genuchten relation directly
        pw_from_sw = v_alpha * ( (se) ** (-v_gamma) - c1 ) ** (v_beta)

    pw_from_sw = -pw_from_sw
    return pw_from_sw


new_sat = van_genuchten_sat_from_pressure(-0.22)
part_p = pressure_from_saturation(new_sat)
np.float64(part_p) ** np.float64(-5.67)    
-0.99999999992010191 ** -5.67

def head_from_sat(sat, z):    
    return van_genuchten_pressure_from_sat(sat) + z
    

head_list = np.apply_along_axis(head_from_sat, 0, initial_sat_3D_reshape.flatten('F'))
np.savetxt('Initial_head.dat', head_list, delimiter=' ')


