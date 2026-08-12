#!/bin/bash

function plotvtk(){
    python3 -m jorek_tools.vtk.plot_vtk "$@"
}