def intersect_linesegment_z0(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Get R-coordinate of intersection between midplane and line segment p2-p1
    """
    p1x, p1y = p1
    p2x, p2y = p2

    return p1x - p1y*(p2x-p1x)/(p2y-p1y)


class MPLoc(Enum):
    inboard = 1
    outboard = 2
def get_midplane_intersections(surface: PostprocProfile, 
                               loc: MPLoc = MPLoc.outboard) -> float:
    """
    Get R-value of the intersection between a surface and the
    midplane.
    """
    centroid_R = np.mean(surface.x_values)
    centroid_Z = np.mean(surface.y_values)

    if loc==MPLoc.outboard:
        theta0 = 0.0
    elif loc==MPLoc.inboard:
        theta0 = np.pi/2.0
    else:
        raise ValueError("Must specify inboard or outboard")

    thetas = np.atan2(
        surface.y_values-centroid_Z,
        surface.x_values-centroid_R
    ) - theta0
    negative_branch = thetas[thetas < 0.0]
    positive_branch = thetas[thetas >= 0.0]

    closest_positive = np.argmin(positive_branch)
    closest_negative = np.argmin(negative_branch)
    
    p1 = (
        surface.x_values[closest_positive],
        surface.y_values[closest_positive]
    )
    p2 = (
        surface.x_values[closest_negative],
        surface.y_values[closest_negative]
    )
    r_midplane = intersect_linesegment_z0(p1, p2)

    return r_midplane