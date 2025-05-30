
def central_density_si(jorek_central_mass: float,
                       jorek_central_number_density: float) -> float:
    """
    Convert JOREK central mass and number density to 
    a central mass density in SI units.

    See https://www.jorek.eu/wiki/doku.php?id=normalization

    :param jorek_central_mass: JOREK central mass [JOREK units]
    :param jorek_central_number_density: JOREK central number density [JOREK units]

    :return: Central mass density in SI units
    """
    m_proton = 1.6726e-27
    n0_normalisation = 1e20
    
    rho0 = jorek_central_mass * \
        jorek_central_number_density * \
            n0_normalisation * \
                m_proton

    return rho0