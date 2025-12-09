#!/usr/bin/python
from argparse import ArgumentParser
#import poincare as pnc
import readxtor as rx
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class XTORProfiles():
    r_maj: np.array
    p_i: np.array
    p_e: np.array
    j_phi: np.array
    #q: np.array

def get_equil_rmaj_profiles(xtfields: str):
    """
    Get equilibrium R_major profiles at Z=0
    """
    iphi=0
    bigr,bigrh,bigz,bigzh = rx.ReadBigrBigz(xtfields)
    p_i = rx.ReadField(xtfields,'Ion_Pressure')[:,:,iphi]
    p_e = rx.ReadField(xtfields,'Electron_Pressure')[:,:,iphi]
    j_phi = rx.ReadField(xtfields,'j')[:,:,iphi,2]


    # bigR, bigZ arrays are given in terms of s,theta co-ordinates.
    # theta = 0 corresponds to index 0,-1 along the second array
    # dimension. theta=pi corresponds to index bigR/Z.shape[1]//2.
    # Some overlap between the inboard and outboard profiles (2 elements
    # to be exact). So, remove these overlapping elements.
    r_maj_outboard = bigr[2:,0]
    p_i_outboard = p_i[2:,0]
    p_e_outboard = p_e[2:,0]
    j_phi_outboard = j_phi[2:,0]

    # R decreases as s increases for theta=pi. Hence, need to reverse
    # the data along the primary axis. 
    n_mid = bigr.shape[1]//2
    r_maj_inboard = bigr[::-1, n_mid]
    p_i_inboard = p_i[::-1, n_mid]
    p_e_inboard = p_e[::-1, n_mid]
    j_phi_inboard = j_phi[::-1, n_mid]

    print(j_phi_inboard)

    return XTORProfiles(
        np.concat((r_maj_inboard, r_maj_outboard)),
        np.concat((p_i_inboard, p_i_outboard)),
        np.concat((p_e_inboard, p_e_outboard)),
        np.concat((j_phi_inboard, j_phi_outboard))
    )
    


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'xtor_filename', type=str, help='Path to XTOR xtfields filename'
    )

    args = parser.parse_args()

    profs = get_equil_rmaj_profiles(args.xtor_filename)
    
    fig, ax = plt.subplots(3)
    ax[0].plot(profs.r_maj, profs.p_e)
    ax[1].plot(profs.r_maj, profs.p_i)
    ax[2].plot(profs.r_maj, profs.j_phi)

    plt.show()
