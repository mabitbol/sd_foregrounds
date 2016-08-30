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

PIXIE_freq_min = 37.5e9 #central frequency of lowest channel (lower bound is 30 GHz)
PIXIE_freq_max = 6.0225e12 #central frequency of highest channel (chose this to get 400 total channels)
PIXIE_freqstep = 15.0e9

# 1 Jy = 1e-26 W / (Hz sr m^2)
# frequencies are expected in Hz

# Kelvin to W/Hz/sr/m^2
# convert from K_rj to spectral radiance in W/Hz/sr/m^2
def krj_to_radiance(nu, y):
    return 2.0 * nu*nu /(clight**2) * kboltz * y

# W/Hz/sr/m^2 to Kelvin
def radiance_to_krj(nu, y):
    return y * clight**2 / (2. * kboltz * nu*nu)

#blackbody T to W/Hz/sr/m^2
def blackbody(nu, T):
    X  = hplanck * nu / (kboltz * T)
    return 2.0 * hplanck * (nu*nu*nu) / (clight**2) * (1.0 / (np.exp(X) - 1.0))

def dbdt(nu, T):
    return 2.0 * (X*X*X*X) * np.exp(X) * (kboltz*T)**3 / (hplanck * clight)**2 / (np.exp(X) - 1.0)**2

# CMB rms in brightness temp
def cmb(freqs, T=TCMB, A=3.0e-6):
    X = hplanck*freqs/(kboltz*T)
    gf = (np.exp(X)-1)**2 / (X*X*np.exp(X))
    return A/gf


# UNITS ARE KELVIN
### Foreground components from PlanckX2015 ###
# see Table 4 of https://arxiv.org/pdf/1502.01588v2.pdf
# Here we are in brightness tempearture (as a first pass) with unit K Rayleigh Jeans
# I've put the best fit Planck values as defaults

# Thermal Dust
# Params Ad, Bd, Td which are amplitude [K_RJ, brightness temp fluctuation w.r.t. CMB blackbody], spectral index, and temperature [K]
# Params were 163e-6, 1.51, 21 but to match jens and some papers we use:
def thermal_dust(nu, Ad=10.e-6, Bd=1.59, Td=19.6):
    nu0 = 545.0e9   #planck frequency
    gam = hplanck/(kboltz*Td)   
    return Ad * (nu/nu0)**(Bd+1.0) * (np.exp(gam*nu0) - 1.0) / (np.exp(gam*nu) - 1.0)

# Synchrotron (based on Haslam and GALPROP) 
# Params As, alpha : amplitude [K_RJ, brightness temp fluctuation w.r.t. CMB blackbody] and shift parameter
# planck says As=20 but matching to Jens gives As~=10.
def synchrotron(nu, As=10.0, alpha=0.26):
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
    return As * (nu0/nu)**2 * numer_fs / denom_fs

# Free-free 
# Params EM, Te : emission measure (=integrated square electron density along LOS) and electron temp [K]
def freefree(nu, EM=13.0, Te=7000.0):
    T4 = (Te * 10**-4)**(-3./2.)
    f9 = nu / (10**9)
    gff = np.log(np.exp(5.960 - (np.sqrt(3.)/np.pi) * np.log(f9*T4)) + np.e)
    tau = 0.05468 * (Te**(-3./2.)) * EM * gff / f9**2
    return (1.0 - np.exp(-tau)) * Te

def freefree2(freqs, EM=100., Te=8000.):
    nu = freqs*1.e-9
    gff = np.log(4.955e-2 / nu) + 1.5 * np.log(Te)
    tff = 3.014e-2 * (Te**-1.5) * (nu**-2) * EM * gff
    return Te * (1. - np.exp(-tff))

# AME
# Params Asd, fp : amplitude [K_RJ, brightness temp fluctuation w.r.t. CMB blackbody] and peak frequency
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
# including this as a check but is of course identical to y-distortion
def sz(nu, ysz=1.4e-6):
    X = hplanck*nu/(kboltz*TCMB)
    gf = (np.exp(X)-1)**2 / (X*X*np.exp(X))
    return ysz*TCMB * ( X*(np.exp(X)+1.)/(np.exp(X)-1.) - 4.) / gf #JCH: fixed some errors here

# CIB
# params TCIB, kf
# units are Jy / sr!!
def cib_jy(nu, TCIB=18.5, KF=0.64):
    X = hplanck*nu/(kboltz*TCIB)
    nu0 = 3.e12
    return 173.4 * TCIB**3 * (nu/nu0)**KF * X**3 / (np.exp(X) - 1.)

def cib(nu, TCIB=18.5, KF=0.64):
    return radiance_to_krj(nu, cib_jy(nu, TCIB, KF)*1e-26)

# CO Line emission



def pixie_frequencies(fmin=PIXIE_freq_min, fmax=PIXIE_freq_max, fstep=PIXIE_freqstep): # PIXIE frequency channels (all in Hz) -- see http://arxiv.org/abs/1105.2044
    PIXIE_freqs = np.arange(fmin, fmax + fstep, fstep)
    return PIXIE_freqs

def pixie_noise(PIXIE_freqs):
    PIXIE_Nfreqs = len(PIXIE_freqs)
    PIXIE_noise = 5.0e-26*np.ones(PIXIE_Nfreqs) #http://arxiv.org/abs/1105.2044 ; taken to be uncorrelated between channels; units W/m^2/Hz/sr 
    return PIXIE_noise
