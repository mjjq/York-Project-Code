#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:52:20 2024

@author: marcus
"""

import numpy as np

@np.vectorize
def jorek_to_alfven_time(times: np.array, B0: float, R0: float) -> np.array:
    """
    Convert JOREK time to Alfven time via normalization schemes for each.
    
    See https://www.jorek.eu/wiki/doku.php?id=normalization
    
    t_JOREK = t_SI / sqrt(mu0*rho0)
    
    t_Alfven = t_SI * (B0/R0) / sqrt(mu0*rho0)
    
    Implies 
    
    t_Alfven = t_JOREK * R0/B0

    Parameters
    ----------
    times : np.array
        JOREK times.
    B0 : float
        Equilibrium magnetic field of plasma.
    R0 : float
        Major radius of the tokamak.

    Returns
    -------
    Times converted from JOREK units to alfven units.

    """
    
    return (R0/B0) * times

@np.vectorize
def jorek_to_alfven_growth(growths: np.array, B0: float, R0: float) -> np.array:
    """
    Convert JOREK time to Alfven growth rate via normalization schemes for each.
    
    See https://www.jorek.eu/wiki/doku.php?id=normalization
    
    t_JOREK = t_SI / sqrt(mu0*rho0)
    
    t_Alfven = t_SI * (B0/R0) / sqrt(mu0*rho0)
    
    Implies 
    
    t_Alfven = t_JOREK * R0/B0
    
    Implies gamma_Alfven = gamma_JOREK * B0/R0

    Parameters
    ----------
    growths : np.array
        JOREK growths.
    B0 : float
        Equilibrium magnetic field of plasma.
    R0 : float
        Major radius of the tokamak.

    Returns
    -------
    Growths converted from JOREK units to alfven units.

    """
    
    return (B0/R0) * growths