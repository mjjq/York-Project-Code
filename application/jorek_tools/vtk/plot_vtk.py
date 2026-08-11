import vtk
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.tri as mtri
from typing import Tuple, List, Optional

def read_vtk_scalar(filename: str, scalar_name: str) -> vtk.vtkUnstructuredGrid:
    # Read VTK file
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.SetScalarsName(scalar_name)
    reader.Update()

    return reader.GetOutput()

def get_min_max(grid: vtk.vtkUnstructuredGrid) -> Tuple[float, float]:
    # Get scalar values
    point_data = grid.GetPointData()

    scalars = point_data.GetScalars()

    vals = vtk_to_numpy(scalars)

    return min(vals), max(vals)

def plot_vtk_heatmap(grid: vtk.vtkUnstructuredGrid,
                     plot_name: str=None,
                     min_max: Tuple[float, float] = None,
                     ax=None):
    # Point coordinates
    points = vtk_to_numpy(grid.GetPoints().GetData())
    x = points[:, 0]
    y = points[:, 1]

    # Get scalar values
    point_data = grid.GetPointData()

    scalars = point_data.GetScalars()

    if scalars is None:
        raise ValueError("No point-data scalar array found.")

    z = vtk_to_numpy(scalars)

    # -------------------------
    # Convert cells to triangles
    # -------------------------
    triangulator = vtk.vtkDataSetTriangleFilter()
    triangulator.SetInputData(grid)
    triangulator.Update()

    tri_grid = triangulator.GetOutput()

    # -------------------------
    # Extract triangle connectivity
    # -------------------------
    triangles = np.empty(
        (tri_grid.GetNumberOfCells(), 3),
        dtype=np.int64
    )

    for i in range(tri_grid.GetNumberOfCells()):
        cell = tri_grid.GetCell(i)

        if cell.GetNumberOfPoints() != 3:
            raise RuntimeError(
                f"Cell {i} has {cell.GetNumberOfPoints()} points"
            )

        for j in range(3):
            triangles[i, j] = cell.GetPointId(j)

    # -------------------------
    # Plot
    # -------------------------
    triang = mtri.Triangulation(x, y, triangles)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5), constrained_layout=True)

    vmin, vmax = None, None
    if min_max:
        vmin, vmax = min_max
    pcm = ax.tripcolor(
        triang,
        z,
        shading="flat",
        cmap="jet",
        vmin=vmin,
        vmax=vmax
    )

    #fig.colorbar(pcm, ax=ax, label=plot_name)

    ax.set_aspect("equal")


    field_data = grid.GetFieldData()
    time = 1000.0*vtk_to_numpy(field_data.GetArray("TIME"))[0]
    ax.set_title(f"Time: {time:.3f}ms")

    # plt.tight_layout()
    # plt.show()

    return pcm

def get_array_names(vtk_filename: str):    
    # Read VTK file
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_filename)
    reader.ReadAllScalarsOn()
    #reader.ReadAllVectorsOn()
    reader.Update()

    data = reader.GetOutput()

    #print(data)

    # Get scalar values
    point_data = data.GetPointData()
    #print(point_data)
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        print(i, array.GetName(), array.GetNumberOfComponents())

def plot_array_of_vtks(grids: List[vtk.vtkUnstructuredGrid], 
                       nrows: int = 1,
                       norm_to: Optional[int] = None):
    ncols = int(np.ceil(len(grids)/nrows))

    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    if len(grids)==1:
        axs = [axs]

    # Normalising data
    min_val, max_val = None, None
    if norm_to is None:
        for i,grid in enumerate(grids):
            min_v, max_v = get_min_max(grid)

            if not min_val:
                min_val = min_v
            else:
                if min_v < min_val:
                    min_val = min_v
        
            if not max_val:
                max_val = max_v
            else:
                if max_v > max_val:
                    max_val = max_v
    else:
        grid = grids[norm_to]
        min_val, max_val = get_min_max(grid)

    for i,grid in enumerate(grids):
        pcm = plot_vtk_heatmap(grid, min_max = (min_val, max_val), ax=axs[i])

    fig.colorbar(pcm, ax=axs[-1])

    
    fig.subplots_adjust(wspace=0, hspace=0)

if __name__=='__main__':
    import sys
    val = sys.argv[1]
    fnames = sys.argv[2:]
    
    grids = [
        read_vtk_scalar(fname, val) for fname in fnames
    ]

    plot_array_of_vtks(grids, norm_to=0)

    plt.show()