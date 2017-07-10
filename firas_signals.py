import numpy as np
from astropy.io import fits
from numpy import log10
from scipy import interpolate

# constants (MKS units, except electron rest mass)
TCMB = 2.725  # Kelvin
hplanck = 6.626068e-34  # MKS
kboltz = 1.3806503e-23  # MKS
clight = 299792458.0  # MKS
m_elec = 510.999  # keV!
jy = 1.e26

def blackbody(nu, T=TCMB):
    X = hplanck * nu / (kboltz * T)
    return jy * 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(X) - 1.0))

def thermal_dust_rad(nu, Ad=1.36e6, Bd=1.53, Td=21.):
    X = hplanck * nu / (kboltz * Td)
    return Ad * X**Bd * X**3. / (np.exp(X) - 1.0)

def DeltaI_mu(freqs, mu_amp=2.e-8):
    X = hplanck*freqs/(kboltz*TCMB)
    return jy * mu_amp * (X / 2.1923 - 1.0)/X * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0

def DeltaI_y(freqs, y_amp=1.77e-6):
    X = hplanck*freqs/(kboltz*TCMB)
    return jy * y_amp * (X / np.tanh(X/2.0) - 4.0) * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0

def firas():
    x = np.loadtxt('data/firas_monopole_spec_v1.txt')
    ffs = x[:, 0] + 0.
    onesigmas = x[:, 3] * 1000   #Jy/sr
    ffs *= clight * 100.         #Hz
    return ffs, onesigmas

def firas2():
    z = np.loadtxt('templates/firassensitivity.txt')
    firasfreqs = z[0]*1.e9
    firasnoise = z[1]
    return firasfreqs, firasnoise

def newmu(nu, mu=-1.e-5):
    X = hplanck*nu/(kboltz*TCMB)
    return -2. * jy * hplanck * nu**3. / clight * np.exp(X+mu) / (np.exp(X+mu) - 1.)**2
