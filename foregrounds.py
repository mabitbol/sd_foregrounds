import numpy as np
from numpy import log10
from scipy import interpolate

hplanck = 6.626068e-34  # MKS
kboltz = 1.3806503e-23  # MKS

def jens_synch_rad(nu, As=288., alps=-0.82, w2s=0.2):
    nu0s = 100.e9
    return As * (nu / nu0s) ** alps * (1. + 0.5 * w2s * np.log(nu / nu0s) ** 2) * 1e-26

def jens_freefree_rad(nu, EM=300.):
    Te = 7000.
    Teff = (Te / 1.e3) ** (3. / 2)
    nuff = 255.33e9 * Teff
    gff = 1. + np.log(1. + (nuff / nu) ** (np.sqrt(3) / np.pi))
    return EM * gff * 1e-26

def spinning_dust(nu, Asd=1.):
    ame_file = np.loadtxt('templates/ame.txt')
    ame_nu = ame_file[0]
    ame_I = ame_file[1]
    fsd = interpolate.interp1d(log10(ame_nu), log10(ame_I), bounds_error=False, fill_value="extrapolate")
    return Asd * 10.**fsd(log10(nu)) 

def thermal_dust_rad(nu, Ad=5.e-26, Bd=1.53, Td=21.):
    nu0 = 545.0e9  # planck frequency
    gam = hplanck / (kboltz * Td)
    return Ad * (nu/1.e9)**2 * (nu / nu0) ** (Bd + 1.0) * (np.exp(gam * nu0) - 1.0) / (np.exp(gam * nu) - 1.0)

def cib_rad(nu, Acib=1.38e-26, Bcib=0.86, Tcib=18.8):
    nu0 = 545.0e9
    gam = hplanck / (kboltz * Tcib)
    return Acib * (nu/1.e9)**2 * (nu / nu0) ** (Bcib + 1.0) * (np.exp(gam * nu0) - 1.0) / (np.exp(gam * nu) - 1.0)

def co_rad(nu, amp=1.):
    x = np.load('templates/co_arrays.npy')
    freqs = x[0]
    co = x[1]
    fs = interpolate.interp1d(log10(freqs), log10(co), bounds_error=False, fill_value="extrapolate")
    return amp * 10. ** fs(log10(nu)) * 1e-26

