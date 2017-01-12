import numpy as np
from astropy.io import fits
from numpy import log10
from scipy import interpolate

# constants (MKS units, except electron rest mass)
TCMB = 2.726  # Kelvin
hplanck = 6.626068e-34  # MKS
kboltz = 1.3806503e-23  # MKS
clight = 299792458.0  # MKS
m_elec = 510.999  # keV!

# Foreground units are in Kelvin
# 1 Jy = 1e-26 W / (Hz sr m^2)
# frequencies are expected in Hz

# Kelvin to W/Hz/sr/m^2
def krj_to_radiance(nu, y):
    return 2.0 * nu * nu / (clight ** 2) * kboltz * y


# W/Hz/sr/m^2 to Kelvin
def radiance_to_krj(nu, y):
    return y * clight ** 2 / (2. * kboltz * nu * nu)


### Reorganizing ###
# These are all actually in W/Hz/sr/m^2!! Jy/sr and then mult by 1e-26
# because the signals in components are in W/Hz/sr/m^2
def jens_synch_rad(nu, As=288., alps=-0.82, w2s=0.2):
    nu0s = 100.e9
    return As * (nu / nu0s) ** alps * (1. + 0.5 * w2s * np.log(nu / nu0s) ** 2) * 1e-26

def jens_synch_rad1(nu, As=288., alps=-0.82):
    nu0s = 100.e9
    return As * (nu / nu0s) ** alps * 1e-26

def jens_synch(nu, As=288., alps=-0.82, w2s=0.2):
    return radiance_to_krj(nu, jens_synch_rad(nu, As, alps, w2s))


def jens_freefree_rad(nu, EM=300.):
    Te = 7000.
    Teff = (Te / 1.e3) ** (3. / 2)
    nuff = 255.33e9 * Teff
    gff = 1. + np.log(1. + (nuff / nu) ** (np.sqrt(3) / np.pi))
    return EM * gff * 1e-26

def jens_freefree(nu, EM=300., Te=7000.):
    return radiance_to_krj(nu, jens_freefree_rad(nu, EM, Te))

def jens_freefree1p(nu, EM=300.):
    return radiance_to_krj(nu, jens_freefree_rad(nu, EM))


def kspinning_dust(nu, Asd=92.e-6):
    nup = 19.0e9
    nu0 = 22.8e9
    nup0 = 30.e9
    ame_file = np.load('templates/spinningdust_template.npy')
    ame_nu = ame_file[0]
    ame_I = ame_file[1]
    fsd = interpolate.interp1d(log10(ame_nu), log10(ame_I), bounds_error=False, fill_value="extrapolate")
    numer_fsd = 10.0 ** fsd(log10(nu * nup0 / nup))
    denom_fsd = 10.0 ** fsd(log10(nu0 * nup0 / nup))
    return Asd * (nu0 / nu) ** 2 * numer_fsd / denom_fsd


def spinning_dust(nu, Asd=1.):
    ame_file = np.loadtxt('templates/ame.txt')
    ame_nu = ame_file[0]
    ame_I = ame_file[1]
    fsd = interpolate.interp1d(log10(ame_nu), log10(ame_I), bounds_error=False, fill_value="extrapolate")
    return Asd * 10.**fsd(log10(nu)) 

def spinning_dust_rad_old(nu, Asd=1470.e-26):
    nup = 19.0e9
    nu0 = 22.8e9
    nup0 = 30.e9
    ame_file = np.load('templates/spinningdust_template.npy')
    ame_nu = ame_file[0]
    ame_I = ame_file[1]
    fsd = interpolate.interp1d(log10(ame_nu), log10(ame_I), bounds_error=False, fill_value="extrapolate")
    numer_fsd = 10.0 ** fsd(log10(nu * nup0 / nup))
    denom_fsd = 10.0 ** fsd(log10(nu0 * nup0 / nup))
    return Asd * numer_fsd / denom_fsd

def spinning_dust2(nu, Asd=92.e-6, nup=19.e9):
    nu0 = 22.8e9
    nup0 = 30.e9
    ame_file = np.load('templates/spinningdust_template.npy')
    ame_nu = ame_file[0]
    ame_I = ame_file[1]
    fsd = interpolate.interp1d(log10(ame_nu), log10(ame_I), bounds_error=False, fill_value="extrapolate")
    numer_fsd = 10.0 ** fsd(log10(nu * nup0 / nup))
    denom_fsd = 10.0 ** fsd(log10(nu0 * nup0 / nup))
    return krj_to_radiance(nu, Asd * (nu0 / nu) ** 2 * numer_fsd / denom_fsd)


def thermal_dust_rad(nu, Ad=5.e-26, Bd=1.53, Td=21.):
    nu0 = 545.0e9  # planck frequency
    gam = hplanck / (kboltz * Td)
    return Ad * (nu/1.e9)**2 * (nu / nu0) ** (Bd + 1.0) * (np.exp(gam * nu0) - 1.0) / (np.exp(gam * nu) - 1.0)

def thermal_dust_rad_old(nu, Ad=163.e-6, Bd=1.53, Td=21.):
    return krj_to_radiance(nu, thermal_dust(nu, Ad, Bd, Td))

def thermal_dust(nu, Ad=163.e-6, Bd=1.53, Td=21.):
    nu0 = 545.0e9  # planck frequency
    gam = hplanck / (kboltz * Td)
    return Ad * (nu / nu0) ** (Bd + 1.0) * (np.exp(gam * nu0) - 1.0) / (np.exp(gam * nu) - 1.0)

def cib_rad(nu, Acib=1.38e-26, Bcib=0.86, Tcib=18.8):
    nu0 = 545.0e9
    gam = hplanck / (kboltz * Tcib)
    return Acib * (nu/1.e9)**2 * (nu / nu0) ** (Bcib + 1.0) * (np.exp(gam * nu0) - 1.0) / (np.exp(gam * nu) - 1.0)

def cib_rad_old(nu, Acib=45.e-6, Bcib=0.86, Tcib=18.8):
    return krj_to_radiance(nu, cib(nu, Acib, Bcib, Tcib))

def cib(nu, Acib=45.e-6, Bcib=0.86, Tcib=18.8):
    nu0 = 545.0e9
    gam = hplanck / (kboltz * Tcib)
    return Acib * (nu / nu0) ** (Bcib + 1.0) * (np.exp(gam * nu0) - 1.0) / (np.exp(gam * nu) - 1.0)


def co_rad(nu, amp=1.):
    x = np.load('templates/co_arrays.npy')
    freqs = x[0]
    co = x[1]
    fs = interpolate.interp1d(log10(freqs), log10(co), bounds_error=False, fill_value="extrapolate")
    return amp * 10. ** fs(log10(nu)) * 1e-26

def co(nu, amp=1.):
    return radiance_to_krj(nu, co_rad(nu, amp))


### Older Functions ###
# blackbody T to W/Hz/sr/m^2
def blackbody(nu, T):
    X = hplanck * nu / (kboltz * T)
    return 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(X) - 1.0))


def dbdt(nu, T):
    return 2.0 * (X * X * X * X) * np.exp(X) * (kboltz * T) ** 3 / (hplanck * clight) ** 2 / (np.exp(X) - 1.0) ** 2


# CMB rms in brightness temp
def cmb(freqs, T=TCMB, A=3.0e-6):
    X = hplanck * freqs / (kboltz * T)
    gf = (np.exp(X) - 1) ** 2 / (X * X * np.exp(X))
    return A / gf


# UNITS ARE KELVIN
### Foreground components from PlanckX2015 ###
# see Table 4 of https://arxiv.org/pdf/1502.01588v2.pdf
# Here we are in brightness tempearture (as a first pass) with unit K Rayleigh Jeans
# I've put the best fit Planck values as defaults

# Thermal Dust
# Params Ad, Bd, Td which are amplitude [K_RJ, brightness temp fluctuation w.r.t. CMB blackbody], spectral index, and temperature [K]
# Params were 163e-6, 1.51, 21 but to match jens and some papers we use:


# Synchrotron (based on Haslam and GALPROP)
# Params As, alpha : amplitude [K_RJ, brightness temp fluctuation w.r.t. CMB blackbody] and shift parameter
# planck says As=20 but matching to Jens gives As~=10.
def synchrotron(nu, As=10.0, alpha=0.26):
    # for details use synch_temp.info and synch_temp[2].columns
    # frequency is in GHz in the file and ranges from 1 MHz to 100 THz
    # spectral radiance is in the next field
    # interpolate to other frequencies
    # interp will throw an error if we give it frequencies outside of the range
    nu0 = 408.0e6  # Hz
    synch_temp = fits.open('templates/COM_CompMap_Synchrotron-commander_0256_R2.00.fits')
    synch_nu = synch_temp[2].data.field(0)  # GHz
    synch_nu *= 1.e9  # Hz
    synch_I = synch_temp[2].data.field(1)  # W/Hz/sr/m^2
    fs = interpolate.interp1d(log10(synch_nu), log10(synch_I))
    numer_fs = 10.0 ** fs(log10(nu / alpha))
    denom_fs = 10.0 ** fs(log10(nu0 / alpha))
    return As * (nu0 / nu) ** 2 * numer_fs / denom_fs


# Free-free
# Params EM, Te : emission measure (=integrated square electron density along LOS) and electron temp [K]
def freefree(nu, EM=15, Te=7000.0):
    T4 = (Te * 10 ** -4) ** (-3. / 2.)
    f9 = nu / (10 ** 9)
    gff = np.log(np.exp(5.960 - (np.sqrt(3.) / np.pi) * np.log(f9 * T4)) + np.e)
    tau = 0.05468 * (Te ** (-3. / 2.)) * EM * gff / f9 ** 2
    return (1.0 - np.exp(-tau)) * Te


def freefree2(freqs, EM=9., Te=7000.):
    nu = freqs * 1.e-9
    gff = np.log(4.955e-2 / nu) + 1.5 * np.log(Te)
    tff = 3.014e-2 * (Te ** -1.5) * (nu ** -2) * EM * gff
    return Te * (1. - np.exp(-tff))


# AME
# Params Asd, fp : amplitude [K_RJ, brightness temp fluctuation w.r.t. CMB blackbody] and peak frequency
def ame(nu, Asd=1.e-4):
    # template nu go from 50 MHz to 500 GHz...
    # had to add a fill value of 1.e-6 at high frequencies...
    nup = 19.0e9
    nu0 = 22.8e9
    nup0 = 33.e9
    ame_temp = fits.open('templates/COM_CompMap_AME-commander_0256_R2.00.fits')
    ame_nu = ame_temp[3].data.field(0)
    ame_nu *= 1.e9
    ame_I = ame_temp[3].data.field(1)
    ame_I /= 1.0e26
    fsd = interpolate.interp1d(log10(ame_nu), log10(ame_I), bounds_error=False, fill_value=-52.5)
    numer_fsd = 10.0 ** fsd(log10(nu * nup0 / nup))
    denom_fsd = 10.0 ** fsd(log10(nu0 * nup0 / nup))
    return Asd * (nu0 / nu) ** 2 * numer_fsd / denom_fsd


def ame2(nu, Asd=92.e-6, nup=19.e9, nu0=22.e9, nup0=30.e9):
    ame_temp = fits.open('templates/COM_CompMap_AME-commander_0256_R2.00.fits')
    ame_nu = ame_temp[3].data.field(0)
    ame_nu *= 1.e9  # Hz
    ame_I = ame_temp[3].data.field(1)  # Jy cm^2 /sr/H
    ame_I /= 1.0e26
    fsd = interpolate.interp1d(log10(ame_nu), log10(ame_I), bounds_error=False, fill_value=-52.5)
    numer_fsd = 10.0 ** fsd(log10(nu * nup0 / nup))
    denom_fsd = 10.0 ** fsd(log10(nu0 * nup0 / nup))
    return Asd * (nu0 / nu) ** 2 * numer_fsd / denom_fsd


# SZ
# params Asz>0
# including this as a check but is of course identical to y-distortion
def sz(nu, ysz=1.4e-6):
    X = hplanck * nu / (kboltz * TCMB)
    gf = (np.exp(X) - 1) ** 2 / (X * X * np.exp(X))
    return ysz * TCMB * (X * (np.exp(X) + 1.) / (np.exp(X) - 1.) - 4.) / gf  # JCH: fixed some errors here


def cib_rad_old(nu, Ambb=170., TCIB=18.5, KF=0.75):
    X = hplanck * nu / (kboltz * TCIB)
    nu0 = 3.e12
    return Ambb * TCIB ** 3 * (nu / nu0) ** KF * X ** 3 / (np.exp(X) - 1.) * 1e-26


def cib_old(nu, Ambb=170., TCIB=18.5, KF=0.75):
    return radiance_to_krj(nu, cib_rad(nu, Ambb, TCIB, KF))
