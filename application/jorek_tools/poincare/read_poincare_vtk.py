import numpy as np
from typing import List
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from jorek_tools.jorek_dat_to_array import PostprocProfile

def read_poincare_vtk(filename: str) -> List[PostprocProfile]:
    """
    Reads poincare data from .vtk (generated using jorek2_connection_fmhd).

    :param filename: Name of the .vtk file

    :return: Data formatted in same way as an array of PostprocProfiles.
    """
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    grid = reader.GetOutput()

    points = vtk_to_numpy(grid.GetPoints().GetData())[:,:2]
    psi_start = vtk_to_numpy(grid.GetPointData().GetArray("psi_start"))
    
    unique_psi_vals = list(set(psi_start))

    ret = []
    for i,psi in enumerate(unique_psi_vals):
        filtered_points = points[psi_start==psi]

        profile = PostprocProfile(
            filtered_points[:,0],
            filtered_points[:,1],
            i
        )

        ret.append(profile)

    return ret

if __name__=='__main__':
    import sys

    fname = sys.argv[1]

    read_poincare_vtk(fname)
