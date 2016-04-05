import numpy as np
import math, os
import glob
from scipy import interpolate
from scipy import integrate
from scipy import special
from astropy.io import fits

# constants (MKS units, except electron rest mass)
TCMB = 2.726 #Kelvin
hplanck=6.626068e-34 #MKS
kboltz=1.3806503e-23 #MKS
clight=299792458.0 #MKS
m_elec = 510.999 #keV!


# convert from uK_rj to spectral radiance in W/Hz/sr/m^2
# frequencies are expected in Hz
def krj_to_radiance(nu, y):
    return 2.0 * nu*nu /(clight**2) * kboltz * y

#blackbody T to W/Hz/sr/m^2
def blackbody(nu, T):
    X  = hplanck * nu / (kboltz * T)
    return 2.0 * hplanck * (nu*nu*nu) / (clight**2) * (1.0 / (np.exp(X) - 1.0))

def dbdt(nu, T):
    return 2.0 * (X*X*X*X) * np.exp(X) * (kboltz*T)**3 / (hplanck * clight)**2 / (np.exp(X) - 1.0)**2


### Foreground components from PlanckX2015 ###
# Here we are in brightness tempearture (as a first pass) with unit K Rayleigh Jeans
# I've put the best fit Planck values as defaults

# Thermal Dust
# Params Ad, Bd, Td which are amplitude, index, and temperature
def thermal_dust(nu, Ad=163.0e-6, Bd=1.51, Td=21.0):
    nu0 = 545.0e9   #planck frequency
    gam = hplanck/(kboltz*Td)   
    return Ad * (nu/nu0)**(Bd+1.0) * (np.exp(gam*nu0) - 1.0) / (np.exp(gam*nu) - 1.0)

# Synchrotron (based on Haslam and GALPROP) 
# Params As, alpha : amplitude and shift parameter
def synchrotron(nu, As=20.0, alpha=0.26):
    #for details use synch_temp.info and synch_temp[2].columns 
    # frequency is in GHz in the file and ranges from 1 MHz to 100 THz
    # spectral radiance is in the next field
    # interpolate to other frequencies
    #interp will throw an error if we give it frequencies outside of the range
    nu0 = 408.0e6                                     # Hz
    synch_temp = fits.open('templates/COM_CompMap_Synchrotron-commander_0256_R2.00.fits')
    synch_nu = synch_temp[2].data.field(0)          # GHz
    synch_nu *= 1.e9                                # Hz
    synch_I = synch_temp[2].data.field(1)           # W/Hz/sr/m^2
    fs = interpolate.interp1d(np.log10(synch_nu), np.log10(synch_I))
    numer_fs = 10.0**fs(np.log10(nu/alpha))
    denom_fs = 10.0**fs(np.log10(nu0/alpha))
    return As * (f0/nu)**2 * numer_fs / denom_fs

# Free-free 
# Params EM, Te : emission measure (=integrated square electron density along LOS) and electron temp
def freefree(nu, EM=13.0, Te=7000.0):
    T4 = (Te * 10**-4)**(-3./2.)
    f9 = nu / (10**9)
    gff = np.log10(np.exp(5.960 - (np.sqrt(3.)/np.pi) * np.log10(f9*T4)) + np.e)
    tau = 0.05468 * (Te**(-3./2.)) * EM * gff / f9**2
    return (1.0 - np.exp(-tau)) * Te

# AME
# Params Asd, fp : amplitude and peak frequency
def ame(nu, Asd=92.0e-6, nup=19.0e9):
    # template nu go from 50 MHz to 500 GHz...
    # had to add a fill value of 1.e-6 at high frequencies...
    nu0 = 22.8e9
    nup0 = 33.35e9
    ame_temp = fits.open('templates/COM_CompMap_AME-commander_0256_R2.00.fits')
    ame_nu = ame_temp[3].data.field(0)
    ame_nu *= 1.e9                      # Hz 
    ame_I = ame_temp[3].data.field(1)   # Jy cm^2 /sr/H
    ame_I *= 1.0e26
    fsd = interpolate.interp1d(np.log10(ame_nu), np.log10(ame_I), bounds_error=False, fill_value=1.e-6)
    numer_fsd = 10.0**fsd(np.log10(nu*nup0/nup))
    denom_fsd = 10.0**fsd(np.log10(nu0*nup0/nup))
    return Asd * (nu0/nu)**2 * numer_fsd / denom_fsd
    
# SZ
# params Asz>0
# including this as a check but shouldnt it be the same as the y distortion 
def sz(nu, ysz=1.4e-6):
    X = hplanck*nu/(kboltz*TCMB)
    gf = (np.exp(X)-1)**2 / (X*X*np.exp(X))
    return (ysz*10**6)*TCMB * ( (X*np.exp(X)+1.)/(np.exp(X)-1.) - 4.) / gf


# Line emission
# this needs more work. should look in paper about CO emission as spectral distortion


# CMB rms in brightness temp
def cmb(freqs, T=TCMB, A=3.0e-6):
    X = hplanck*freqs/(kboltz*T)
    gf = (np.exp(X)-1)**2 / (X*X*np.exp(X))
    return A/gf
    

