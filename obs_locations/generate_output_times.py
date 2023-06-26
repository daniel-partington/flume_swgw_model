import numpy as np

target_times = [0, 1816, 3273, 4800, 7020, 9004, 10850, 12120, 13800, 15200]
divisions = 30

with open('output_times.dat', 'w') as f:

    f.write('output times\n')
    for i in range(len(target_times) - 1):
        for time in np.linspace(target_times[i], target_times[i + 1], 
                                num=divisions, endpoint=False):
            f.write('{}\n'.format(time))
    f.write('end')

target_times = [0] + [float(x) + 7776000.0 for x in [0, 1816, 3273, 4800, 7020, 9004, 10850, 12120, 13800, 15200]]
divisions = 30

with open('output_times_with_drainage.dat', 'w') as f:

    f.write('output times\n')
    for i in range(len(target_times) - 1):
        for time in np.linspace(target_times[i], target_times[i + 1], 
                                num=divisions, endpoint=False):
            f.write('{}\n'.format(time))
    f.write('end')
