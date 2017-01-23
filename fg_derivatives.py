import numpy as np
from numpy import log10
from scipy import interpolate

hplanck = 6.626068e-34  # MKS
kboltz = 1.3806503e-23  # MKS

def d_dust_dA(nu, Ad=1.36e6, Bd=1.53, Td=21.):
    X = hplanck * nu / (kboltz * Td)
    return X**(Bd+3.) / (np.exp(X) - 1.0)
def d_dust_dB(nu, Ad=1.36e6, Bd=1.53, Td=21.):
    X = hplanck * nu / (kboltz * Td)
    return Ad * X**(Bd+3.) * np.log(X)/ (np.exp(X) - 1.0)
def d_dust_dT(nu, Ad=1.36e6, Bd=1.53, Td=21.):
    X = hplanck * nu / (kboltz * Td)
    return (-Ad*X**(Bd+3.)/Td) * ( (Bd+3.)/(np.exp(X)-1.) - X*np.exp(X)/(np.exp(X)-1.)**2 ) 

def d_cib_dA(nu, Acib=3.46e5, Bcib=0.86, Tcib=18.8):
    X = hplanck * nu / (kboltz * Tcib)
    return X**(Bcib+3.) / (np.exp(X) - 1.0)
def d_cib_dB(nu, Acib=3.46e5, Bcib=0.86, Tcib=18.8):
    X = hplanck * nu / (kboltz * Tcib)
    return Acib * X**(Bcib+3.) * np.log(X)/ (np.exp(X) - 1.0)
def d_cib_dT(nu, Acib=3.46e5, Bcib=0.86, Tcib=18.8):
    X = hplanck * nu / (kboltz * Tcib)
    return (-Acib*X**(Bcib+3.)/Tcib) * ( (Bcib+3.)/(np.exp(X)-1.) - X*np.exp(X)/(np.exp(X)-1.)**2 ) 

def d_synch_dA(nu, As=288., alps=-0.82, w2s=0.2):
    nu0 = 100.e9
    return (nu / nu0) ** alps * (1. + 0.5 * w2s * np.log(nu / nu0) ** 2)
def d_synch_dalps(nu, As=288., alps=-0.82, w2s=0.2):
    nu0 = 100.e9
    return As * np.log(nu/nu0) * (nu / nu0) ** alps * (1. + 0.5 * w2s * np.log(nu / nu0) ** 2)
def d_synch_dw2s(nu, As=288., alps=-0.82, w2s=0.2):
    nu0 = 100.e9
    return 0.5 * As * (nu / nu0) ** alps * np.log(nu / nu0) ** 2

def d_freefree_dEM(nu):
    Te = 7000.
    Teff = (Te / 1.e3) ** (3. / 2)
    nuff = 255.33e9 * Teff
    return 1. + np.log(1. + (nuff / nu) ** (np.sqrt(3) / np.pi))

def d_ame_dA(nu):
    ame_file = np.loadtxt('templates/ame.txt')
    ame_nu = ame_file[0]
    ame_I = ame_file[1] * 1.e26
    fsd = interpolate.interp1d(log10(ame_nu), log10(ame_I), bounds_error=False, fill_value="extrapolate")
    return 10.**fsd(log10(nu))

def d_co_dA(nu):
    x = np.load('templates/co_arrays.npy')
    freqs = x[0]
    co = x[1]
    fs = interpolate.interp1d(log10(freqs), log10(co), bounds_error=False, fill_value="extrapolate")
    return 10. ** fs(log10(nu))




