
def generate_grok_for_flume_model(grok_fname,
                                  inflow_data='inflow_experiment_2.dat',
                                  varying_surface=False,
                                  heterogeneous_k=False,
                                  heterogeneous_mannings_n=False):

    with open(grok_fname, 'w') as f:
        f.write(
'''!=================================================='
!============ Problem description =================
!==================================================
Flume model from Water Research Laboratory flume experiments at University of New South Wales
end title

! Length = 8 m
! Depth = 0.6 m
! Width = 0.25 m
! Slope = 0.003

! Free drainage base?

! Potential thin layer retardation of flow through felt at bottom

! ## Sand properties
! n = 0.4
! Thickness of sand 0.45 m
!==================================================
!================ Grid generation =================
!==================================================

generate uniform rectangles
	8.0, 160	
	0.25, 1

	generate layers interactive

	  base elevation
	  	elevation from bilinear function in xy
	    0.0 8.0 0.0 0.25
	    0.0 -0.003 0.0 0.0 0.0

	  end base elevation

''')
        if not varying_surface:
            f.write('''            
	  new layer
	    layer name
	    sand
	      elevation from bilinear function in xy
	    0.0 8.0 0.0 0.25
	    0.45 -0.003 0.0 0.0 0.0
		uniform sublayering
	    45
	  end ! new layer\n''')
        else:
            f.write('''            
        new layer
	    layer name
	    sand
          elevation from bilinear function in xy
	    0.0 8.0 0.0 0.25
	    0.44 -0.003 0.0 0.0 0.0
		uniform sublayering
	    44
	  end ! new layer
   	  new layer
	    layer name
	    sand_top
          elevation from raster file
          riverbed_elevation.dat
        end ! new layer
\n''')
        f.write('''       
	end ! generate layers interactive
end grid generation

mesh to tecplot
flume_mesh.dat

!=======================================
!==== General simulation parameters ====
!=======================================
finite difference mode
unsaturated
transient flow
dual nodes for surface flow
compute underrelaxation factor
units: kilogram-metre-second

!==================================================
!================== SURFACE =======================
!==================================================
! -------- surface flow properties ----------------
use domain type
surface

properties file
./properties/flume_riverbed.oprops

! river faces
clear chosen nodes
clear chosen faces
choose faces top
new zone
    1
clear chosen zones
choose zone number
    1
read properties
    flume
    \n''')

        if heterogeneous_mannings_n:
            f.write('Map isotropic k from raster\nspatial_k.dat\n\n')
        f.write('''    
! ------------- initial conditions ----------------
clear chosen zones
clear chosen faces
clear chosen nodes

choose nodes all
create node set
    surface_nodes

!initial water depth
!   1E-7

! ------------ boundary conditions ----------------
clear chosen zones
clear chosen faces
Clear chosen nodes

choose nodes top
    create face set
    top
write chosen faces
    top_faces

!--- Inflows\n''')
        f.write('include ./input_data/{}'.format(inflow_data))
        f.write('''

!--- critical depth outflow
clear chosen zones
clear chosen faces
clear chosen segments
clear chosen elements
clear chosen nodes

choose node
	8.0 0.0 0.426
choose node
	8.0 0.25 0.426

create segment set
    critical_depth_segment

boundary condition
        type
        critical depth
		name
		crit_depth
        segment set
        critical_depth_segment
		tecplot output
end


!==================================================
!================= POROUS MEDIA ===================
!==================================================
! ------- subsurface properties -------------------
use domain type
porous media

properties file
./properties/material_sand.mprops

! ---- aquifer properties
clear chosen zones
clear chosen faces
clear chosen segments
clear chosen elements
clear chosen nodes

Choose elements all
new zone
    2
clear chosen zones
choose zone number
    2
read properties
    sand\n''')
        if heterogeneous_k:
            f.write('Map isotropic k from raster\nspatial_k.dat\n\n')

        f.write('''    
!  ----  flume riverbed properties 
clear chosen zones
clear chosen nodes
clear chosen elements

new zone
    3
clear chosen zones
choose zone number
    3
read properties
    sand_riverbed


! ------------ initial conditions -----------------
clear chosen zones
clear chosen nodes
clear chosen elements

choose nodes all

!Initial head surface elevation
Function x initial head
0.0  0.0
8.0  -0.024 
! ---------- boundary conditions ------------------
clear chosen zones
clear chosen nodes
clear chosen elements

choose nodes bottom
create node set 
bottom

boundary condition
	type
	head equals elevation

	node set
	bottom
        tecplot output
end ! new specified head

!======================================================
!==================TIMESTEP CONTROLS===================
!======================================================

head control
0.5
saturation control
0.050

maximum timestep
100.
initial timestep
0.00001
maximum timestep multiplier
2.0
minimum timestep multiplier
0.5

!underrelaxation factor
!0.1
compute underrelaxation factor

remove negative coefficients

nodal flow check tolerance
1E-3

newton information

!flow solver maximum iterations
!1000

!======================================================
!=======================OUTPUT=========================
!======================================================

include ./obs_locations/output_times.dat

!==================================================
!============== Observation Points ================
!==================================================

include ./obs_locations/outputs.dat

K to tecplot
''')