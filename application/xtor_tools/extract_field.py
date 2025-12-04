import os
from matplotlib import pyplot as plt
import numpy as np

import readxtor as rx
import analysis as an

def get_rz_coords(xtfields_data):
    lmax,mmax,nmax = rx.lmnSize(xtfields_data[0])
    M = an.Metrics(lmax=lmax,mmax=mmax,nmax=nmax)

    R = M.bigrh
    Z = M.bigzh

    return R, Z

def read_jphi(xtfields_filename: str):
    root, ext = os.path.splitext(xtfields_filename)
    j_field = rx.ReadField(xtfields_filename,'j')

    j_phi = j_field[:,:,0,2]

    return j_phi

def map_field_to_coords():
    lmax = c.shape[0]-1
    mmax = c.shape[1]-1
    theta = np.mod(theta,2.*np.pi)
    it1 = int(rx.cont_2_ind(theta,np.linspace(0,2.*np.pi,mmax+1)))
    it2 = int(np.mod(it1 + mmax/2,mmax))
    drh=1./(2.*lmax)
    sval = np.linspace(0.,1.,lmax+1)
    svalh = np.linspace(-drh,1.-drh,lmax+1)
    # Build data
    if Stheta:
        r = np.zeros(lmax+1)
        data = np.zeros(lmax+1)
        if ih=='h':
            r=svalh
            data[0:lmax] = c[0:lmax,it1]
        elif ih=='i':
            r=sval
            data[0:lmax] = c[1:lmax+1,it1]
        #data[1:lmax+1] = c[1:lmax+1,it1]
    else:
        if ih=='h':
            R0 = 0.5*(x[0,0] + x[1,0])
            Z0 = 0.5*(y[1,1] + y[1,mmax-1])
        elif ih=='i':
            R0 = x[1,0]
            Z0 = y[1,0]
        x = x-R0
        y = y-Z0
        r = np.zeros(2*lmax)
        r[lmax:] = np.sqrt(x[1:lmax+1,it1]**2 + y[1:lmax+1,it1]**2)
        r[:lmax] = -np.sqrt(x[lmax+1:0:-1,it2]**2 + y[lmax+1:0:-1,it2]**2)
        data = np.zeros(2*lmax)
        data[lmax:] = c[1:lmax+1,it1]
        data[:lmax] = sign*c[lmax+1:0:-1,it2]
    # Plot
    return r, data

if __name__=='__main__':
    j = read_jphi('xtfields0')
    
    plt.plot(j[:,0])
    plt.plot(j[:,j.shape[1]//2])
    plt.show()