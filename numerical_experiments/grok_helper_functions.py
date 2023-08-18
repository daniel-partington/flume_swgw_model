import numpy as np

def gen_output_times(fname, flows, period_between_flow_changes):
    with open(fname, 'w') as f:
        f.write('output times\n')
        times = np.arange(0, len(flows) * period_between_flow_changes, period_between_flow_changes)
        np.savetxt(f, times, fmt='%.18e', delimiter='\n')
        f.write('end')
        
def gen_input_for_inflows(fname, flows, period_between_flow_changes, run_name='default run'):
    times = np.arange(0, len(flows) * period_between_flow_changes, period_between_flow_changes)
    time_flow = np.array(zip(times, flows))
    with open(fname, 'w') as f:
        f.write('''
! Inflow to Flume for {}

clear chosen nodes
	choose node
	0.0 0.0 0.6
	choose node
	0.0 0.25 0.6
	create node set
	flume_in

boundary condition
        type
        flux nodal
		node set
		flume_in
		time value table
'''.format(run_name))
        np.savetxt(f, time_flow, fmt='%.18e', delimiter=' ', newline='\n')
        f.write('''
           end
tecplot output
end''')


def gen_raster_top(x, mu, sigma, fname, minx=None, pad=True):
    ''' 
    Generate raster file (for HGS) for flume model using Gaussian distribution
    :param: x: 1D numpy array
    :param: mu: (float) mean of variation in parameter of interest 
    :param: sigma: (float) standard deviation of variation in parameter of interest 
    :param: fname: (str) filename to write raster to 
        
    returns str
    
    Raster format accepted for HGS 
        NCOLS xxx
        NROWS xxx
        XLLCORNER xxx
        YLLCORNER xxx
        CELLSIZE xxx
        NODATA_VALUE xxx
        row 1
        row 2
        ...
        row n
        where NCOLS and NROWS are the number of rows and columns in the raster respectively,
        XLLCORNER and YLLCORNER are the x and y coordinates of the lower left corner of the
        raster, CELLSIZE is the size of each raster cell, NODATA_VALUE is the value in the ASCII file
        representing cells whose true value is unknown (i.e. missing) and xxx is a number. Row 1
        of the data is at the top of the grid, row 2 is just under row 1, and so on.
    '''
    adjustments = np.random.normal(mu, sigma, x.shape[0])
    x_adjust = x - adjustments
    def padlr(arr):
        return np.append(np.insert(arr, 0, arr[0]), arr[-1])
    if pad:
        x_adjust = padlr(x_adjust)
    if minx is not None:
        if pad:
            minx = padlr(minx)
        mask = x_adjust - minx < 0
        x_adjust[mask] = minx[mask] + 0.0001
    np.save(fname.replace('.dat',''), x_adjust)
    cellsize = 0.05
    ncols = 160 + 2
    nrows =  int(0.25 / 0.05) + 2
    xy = np.stack([x_adjust] * nrows)
    xllcorner = 0 - 0.05
    yllcorner = 0 - 0.05
    nodata_value = -999
    with open(fname, 'w') as f:
        f.write('''NCOLS {}
NROWS {}
XLLCORNER {}
YLLCORNER {}
CELLSIZE {}
NODATA_VALUE {}\n'''.format(ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value)
        )
        np.savetxt(f, xy, fmt='%.18e', delimiter=' ', newline='\n')        
        
def gen_oprops(mannings_n=0.024186654284876676, rill_storage_height=0.0001, 
               obstruction_storage_height=0.0, coupling_length=0.001, 
               fname='flume_riverbed.oprops'):

    with open(fname, 'w') as f:
        f.write('''! Flume

!---------- Flume river ------------
!
! Reported Darcy-weisbach friction factor from experiments:
!
!
flume
x friction
{}
y friction
{}
rill storage height
{}
obstruction storage height
{}
coupling length
{}
end ! material'''.format(mannings_n, mannings_n, rill_storage_height, 
                         obstruction_storage_height, coupling_length))    
            
def gen_mprops(k=3.03E-4, porosity=0.4, alpha=12.0539, beta=5.67, 
               residual_saturation=0.05, fname='material_sand.mprops'):

    with open(fname, 'w') as f:
        f.write('''! Flume aquifer

!---------- Sand Flume ------------
sand
k isotropic
   {} 
porosity
{}
unsaturated VAN GENUCHTEN FUNCTIONS
alpha
     {}
beta
     {}
residual saturation
{}
Minimum relative permeability
0.001
table smoothness factor
0.0000001
generate tables from unsaturated functions
end functions
end material

!---------- Riverbed Flume ------------
sand_riverbed
k isotropic
    2.4E-4
porosity
0.4
unsaturated VAN GENUCHTEN FUNCTIONS
alpha
     3.4800000E+00
beta
    1.7500000E+00
residual saturation
0.05
Minimum relative permeability
0.001
table smoothness factor
0.0000001
generate tables from unsaturated functions
end functions
end material'''.format(k, porosity, alpha, beta, residual_saturation))            