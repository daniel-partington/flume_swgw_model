import os
import datetime
import pandas as pd
import numpy as np
import pyemu

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

par_names = ['k', 'n', 'vg_alpha', 'vg_beta', 'rs', 'kappa']

#obs_names = sat_obs.obsnme.tolist()
#Pst = pyemu.pst.pst_utils.generic_pst(par_names=par_names, obs_names=obs_names)
