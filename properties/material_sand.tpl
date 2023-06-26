ptf #
! Flume aquifer

!---------- Sand Flume ------------
sand
k isotropic
    #k# !3.73E-4
porosity
#n#
unsaturated VAN GENUCHTEN FUNCTIONS
alpha
     #vgalph#
beta
     #vgbeta#
residual saturation
#rs#
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
end material