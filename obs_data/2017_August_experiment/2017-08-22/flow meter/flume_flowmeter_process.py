import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime(2017,8,22,12,34,4)

fnames = ["Flowmeter_{}_Flume.CSV".format(x) for x in ['1', '2']]
          
flowmeter_dfs = {}
flowmeter_dfs_resampled = {}
resample_freq = '30S'
cutoff = 13800

for index, fname in enumerate(fnames):
    flowmeter_dfs[index] = pd.read_csv(fname)
    fm_df = flowmeter_dfs[index]
    fm_df['Date/Time'] = pd.to_datetime(fm_df['Date/Time'])
    fm_df.loc[:, 'Elapsed Time'] = [t.total_seconds() for t in fm_df['Date/Time'] - start]
    fm_df.index = fm_df['Date/Time']
    fm_df = fm_df[fm_df['Elapsed Time'] >= 0]
    fm_df
    fm_df_rs = fm_df.copy()
    fm_df_rs.index = fm_df_rs.index - datetime.timedelta(4)
    fm_df_rs_f = fm_df_rs[['Flow Rate', 'Total']].resample(resample_freq).mean()
    fm_df_rs_t = fm_df_rs[['Elapsed Time', 'Total']].resample(resample_freq).max()
    fm_df_rs = fm_df_rs_f.merge(fm_df_rs_t, how='inner', left_index=True, right_index=True).drop(['Total_x', 'Total_y'], axis=1)
    fm_df_rs = fm_df_rs[fm_df_rs['Elapsed Time'] < cutoff]
    flowmeter_dfs_resampled[index] = fm_df_rs

flowmeter_dfs[2] = flowmeter_dfs[1].copy() 
flowmeter_dfs[2]['Flow Rate'] = flowmeter_dfs[0]['Flow Rate'] + flowmeter_dfs[1]['Flow Rate']
fm_df_rs = flowmeter_dfs[2].copy()
fm_df_rs.index = fm_df_rs.index - datetime.timedelta(4)
fm_df_rs_f = fm_df_rs[['Flow Rate', 'Total']].resample(resample_freq).mean()
fm_df_rs_t = fm_df_rs[['Elapsed Time', 'Total']].resample(resample_freq).max()
fm_df_rs = fm_df_rs_f.merge(fm_df_rs_t, how='inner', left_index=True, right_index=True).drop(['Total_x', 'Total_y'], axis=1)
fm_df_rs = fm_df_rs[fm_df_rs['Elapsed Time'] < cutoff]
flowmeter_dfs_resampled[2] = fm_df_rs

fig = plt.figure()          
ax = fig.add_subplot(2, 1, 1)
for index, fname in enumerate(fnames):
    fm_df = flowmeter_dfs[index]
    ax.plot(fm_df['Elapsed Time'], fm_df['Flow Rate'], label='Flowmeter {}'.format(index + 1))
ax.plot(flowmeter_dfs[2]['Elapsed Time'], flowmeter_dfs[2]['Flow Rate'], label='Total')
ax.legend(loc='lower center', fontsize=10, frameon=False)
ax.set_ylabel('Flow [L/min]')
#ax.set_xlabel('Time [s]')
ax.set_xticklabels('')
ax.set_title('Raw flowmeter data')
ax.set_xlim(0, cutoff)

ax = fig.add_subplot(2, 1, 2)
for index, fname in enumerate(fnames):
    fm_df_rs = flowmeter_dfs_resampled[index]
    ax.plot(fm_df_rs['Elapsed Time'], fm_df_rs['Flow Rate'], label='Flowmeter {}'.format(index + 1))
ax.plot(flowmeter_dfs_resampled[2]['Elapsed Time'], flowmeter_dfs_resampled[2]['Flow Rate'], label='Total')
ax.set_ylabel('Flow [L/min]')
ax.set_xlabel('Time [s]')
ax.set_title('Resampled flowmeter data')
ax.set_xlim(0, cutoff)

# Write out to format that HGS grok can interpret:
for index in range(3):
    fm_df_rs = flowmeter_dfs_resampled[index]
    fm_df_rs[['Elapsed Time', 'Flow Rate']].to_csv('flowmeter{}_inflow.txt'.format(index + 1), header=False, index=False)
