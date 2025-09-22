import numpy as np

def grid_gaussian_dist(r0: float,
                       rs: float,
                       thickness: float,
                       resolution: float) -> np.array:
    max_index = np.ceil((1.0/resolution) * np.sqrt(np.log(np.abs(r0-rs)/thickness)))

    linear_grid = np.arange(0, max_index, 1)

    gaussian_grid = rs - (rs-r0)*np.exp(-(linear_grid*resolution)**2)

    gaussian_grid[-1] = rs + np.sign(r0-rs)*thickness

    return gaussian_grid

def grid_uniform_dist(r0: float,
                      rs: float,
                      thickness: float,
                      resolution: float) -> np.array:
    
    grid = np.arange(
        r0, 
        rs+np.sign(r0-rs)*thickness, 
        np.sign(rs-r0)*resolution
    )
    grid[-1] = rs + np.sign(r0-rs)*thickness

    return grid

def grid_uniform_split(rs: float,
                       rs_thickness: float,
                       resolution: float) -> np.array:
    # Convert to number of points since linspace ensures
    # we have points at 0.0 and 1.0
    n_points = int(1.0/resolution)
    
    fullgrid = np.linspace(0.0, 1.0, n_points)


    fwd_grid = fullgrid[fullgrid < rs-rs_thickness]
    bkwd_grid = fullgrid[fullgrid > rs+rs_thickness][::-1]

    fwd_grid[-1] = rs-rs_thickness
    bkwd_grid[-1] = rs+rs_thickness

    return fwd_grid, bkwd_grid

if __name__=='__main__':
    grid_gauss_left = grid_gaussian_dist(
        0.0,
        0.5,
        1e-4,
        1e-3
    )
    print(len(grid_gauss_left))
    print(grid_gauss_left)

    grid_gauss_right = grid_gaussian_dist(
        1.0, 0.5, 1e-4, 1e-3
    )
    print(grid_gauss_right)

    from matplotlib import pyplot as plt
    #plt.scatter([i for i,g in enumerate(grid_gauss_left)], grid_gauss_left)
    #plt.scatter([i for i,g in enumerate(grid_gauss_right)], grid_gauss_right)
    plt.scatter(grid_gauss_left, [0.0]*len(grid_gauss_left))
    plt.show()
