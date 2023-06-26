import numpy as np

slope = -0.003
start_elevation_flume = 0.45
flume_length = 8.0

with open("outputs.dat", 'w') as f:
    
    f.write('''
! ## Observation locations for sensors:
! Horizontal (Sx): 0.5 - 5.5 in 1 m increments
! Depths of (y): 0.05 - 0.4 in increments of 0.05 m
! Naming convention of sensor locations: Sx-y
! Loggers functioned at a maximum resolution of 2 s.
! Usable data:
! S1-8, S2-8, S3-8, S4-8, S1-7, S2-7, S3-7, S4-7, S1-6, S2-6, S4-6, S1-5, S2-5, S3-5, S4-5, S1-4, S2-4, S3-4, S4-4, S1-3, S2-3, S4-3, S1-2, S2-2, S4-2, S1-1, S2-1
    ''')
    
    for xloc, x in enumerate(np.linspace(0.5, 5.5, 6)):
        for zloc, z in enumerate(np.linspace(0.05, 0.4, 8)):
            f.write("\nMake observation point\nS{}-{}\n".format(xloc + 1, zloc + 1))
            f.write("{} 0.0 {}\n\n".format(x, z + x * slope))
            
    f.write('''
\n! Hydrographs:\n  
use domain type
surface\n  
    ''')            

    for x in np.linspace(0.0, flume_length, 81):
        f.write('\n!--- Hydrograph @ x = {}\n'.format(x))
        f.write('clear chosen nodes\n')
        f.write('clear chosen faces\n')
        f.write('choose node\n {} 0.0 {}\n'.format(x, start_elevation_flume + x * slope))
        f.write('choose node\n {} 0.25 {}\n'.format(x, start_elevation_flume + x * slope))
        f.write('Set hydrograph nodes\n_x{}_\n'.format(x))            