import numpy as np
from numpy import log10
from scipy import interpolate

hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS

ndp = np.float128

def synch_curv(nu, As=288., alps=-0.82, w2s=0.2):
    nu0s = 100.e9
    return (As * (nu / nu0s) ** alps * (1. + 0.5 * w2s * np.log(nu / nu0s) ** 2)).astype(ndp)

def synch(nu, As=288., alps=-0.82):
    nu0s = 100.e9
    return (As * (nu / nu0s) ** alps).astype(ndp)

def freefree(nu, EM=300.):
    Te = 7000.
    Teff = (Te / 1.e3) ** (3. / 2)
    nuff = 255.33e9 * Teff
    gff = 1. + np.log(1. + (nuff / nu) ** (np.sqrt(3) / np.pi))
    return (EM * gff).astype(ndp)

def spinning_dust(nu, Asd=1.):
    ame_nu, ame_I = np.load('templates/ame.npy')
    fsd = interpolate.interp1d(log10(ame_nu), log10(ame_I), bounds_error=False, fill_value="extrapolate")
    return (Asd * 10.**fsd(log10(nu)) * 1.e26).astype(ndp)

def thermal_dust(nu, Ad=1.36e6, Bd=1.53, Td=21.):
    X = hplanck * nu / (kboltz * Td)
    return (Ad * X**Bd * X**3. / (np.exp(X) - 1.0)).astype(ndp)

def cib(nu, Acib=3.46e5, Bcib=0.86, Tcib=18.8):
    X = hplanck * nu / (kboltz * Tcib)
    return (Acib * X**Bcib * X**3. / (np.exp(X) - 1.0)).astype(ndp)

def co(nu, Aco=1.):
    freqs, co = np.load('templates/co_arrays.npy')
    fs = interpolate.interp1d(log10(freqs), log10(co), bounds_error=False, fill_value="extrapolate")
    return (Aco * 10.**fs(log10(nu))).astype(ndp)

