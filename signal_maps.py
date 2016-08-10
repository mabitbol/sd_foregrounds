import numpy as np
from numpy import log10
import pylab as pl
import seaborn as sns
import scipy.optimize as scop
from scipy import interpolate
import healpy as hp
from astropy.io import fits
import foregrounds

## everything is returned in MICRO kelvin
## want to return maps scaled to any frequency
## make a function to add them all up (and CMB)
## make a function that can grab a portion of the sky

nup = 10.e9 #Hz
freqs = np.logspace(8, 11, 100)  #100 MHz to 100 GHz

def synch_map(nu=nup):
    syncht = hp.read_map('templates/COM_CompMap_Synchrotron-commander_0256_R2.00.fits')
    synch = foregrounds.synchrotron(freqs)
    fs = interpolate.interp1d(log10(freqs), log10(synch))
    scaling = 10.**fs(log10(nu)) / 10.**fs(log10(408.e6))
    return syncht*scaling

def ff_map(nu=nup):
    freefree = hp.read_map('templates/COM_CompMap_freefree-commander_0256_R2.00.fits')
    free_temp = fits.open('templates/COM_CompMap_freefree-commander_0256_R2.00.fits')
    free_EM = free_temp[1].data.field(1)
    free_T = free_temp[1].data.field(3)
    ff = foregrounds.freefree(nu, free_EM, free_T)
    ffring = hp.reorder(ff, n2r=True)
    return ffring*1.e6

def ame_map(nu=nup):
    spinningdust = hp.read_map('templates/COM_CompMap_AME-commander_0256_R2.00.fits')
    spinningdust2 = hp.read_map('templates/COM_CompMap_AME-commander_0256_R2.00.fits', hdu=2)
    ame = foregrounds.ame(freqs, Asd=92., nup=19.e9, nu0=22.8e9)
    ame2 = foregrounds.ame(freqs, Asd=18., nup=19.e9, nu0=41.e9)
    fs_ame = interpolate.interp1d(log10(freqs), log10(ame))
    fs_ame2 = interpolate.interp1d(log10(freqs), log10(ame2))
    scaling = 10.**fs_ame(log10(nu)) / 10.**fs_ame(log10(22.8e9))
    scaling2 = 10.**fs_ame(log10(nu)) / 10.**fs_ame(log10(41.e9))
    return spinningdust*scaling + spinningdust2*scaling2

def foreground_sky(nu=nup):
    return synch_map(nu) + ff_map(nu) + ame_map(nu)

def cmb_map():
    return hp.read_map('templates/cmb_map256.fits')*1.e6

def total_sky(nu=nup):
    return foreground_sky(nu) + cmb_map() + 2.725e6
