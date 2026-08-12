#!/bin/bash

function plotvtk(){
    python3 -m jorek_tools.vtk.plot_vtk "$@"
}

function plotvtk-pnt(){
    plotvtk "$@" -v Pres -t "Pressure [JU]" -o pressure.png &
    plotvtk "$@" -v Te_keV -t "\$T_e\$ (keV)" -o temperature.png &
    plotvtk "$@" -v ne20_m-3 -t "\$n_e\$ (10\$^{20}\$/m\$^3\$)" -o density.png &
}