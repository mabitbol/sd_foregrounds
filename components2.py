import numpy as np
import math, os
import glob
from scipy import interpolate
from scipy import integrate
from scipy import special
from astropy.io import fits
from foregrounds import radiance_to_krj as r2k

"""
A model for each component signal in the spectral distortion sky.
The function for each component takes as input a vector of frequency values (in Hz), a parameter (or parameters) describing the amplitude,
and (if necessary) other parameters that specify the SED, which we may want to vary (e.g., dust spectral index).
The output is Delta I, the specific intensity measured with respect to the assumed fiducial blackbody spectrum at T_CMB = 2.726 K.
"""

# constants (MKS units, except electron rest mass)
TCMB = 2.726 #Kelvin
hplanck=6.626068e-34 #MKS
kboltz=1.3806503e-23 #MKS
clight=299792458.0 #MKS
m_elec = 510.999 #keV!

# Delta_T distortion (i.e., this accounts for the difference in the true CMB temperature from our assumed value)
#   N.B. I am only working to lowest order here, we may want to go beyond this (see Eq. 2 of http://arxiv.org/pdf/1306.5751v2.pdf and discussion thereafter; also Sec. 4.1 of that paper)
def DeltaI_DeltaT(freqs, DeltaT_amp=1.2e-4): #freqs in Hz, DeltaT_amp dimensionless (DeltaT = (T_CMB_true - T_CMB_assumed)/T_CMB_assumed), DeltaI_DeltaT in W/m^2/Hz/sr
    X = hplanck*freqs/(kboltz*TCMB)
    return DeltaT_amp * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0

def kDeltaI_DeltaT(freqs, DeltaT_amp=1.2e-4): 
    return r2k(freqs, DeltaI_DeltaT(freqs, DeltaT_amp))

# y-type distortion (i.e., non-relativistic tSZ) -- see e.g. Eq. 6 of Hill+2015
def DeltaI_y(freqs, y_amp=1.77e-6): #freqs in Hz, y_amp dimensionless, DeltaI_y in W/m^2/Hz/sr
    X = hplanck*freqs/(kboltz*TCMB)
    return y_amp * (X / np.tanh(X/2.0) - 4.0) * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0

def kDeltaI_y(freqs, y_amp=1.77e-6):
    return r2k(freqs, DeltaI_y(freqs, y_amp))

# mu-type distortion -- see e.g. Section 2.3 of 1306.5751
def DeltaI_mu(freqs, mu_amp=2.e-8): #freqs in Hz, mu_amp dimensionless, DeltaI_mu in W/m^2/Hz/sr
    X = hplanck*freqs/(kboltz*TCMB)
    return mu_amp * (X / 2.1923 - 1.0)/X * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0

def kDeltaI_mu(freqs, mu_amp=2.e-8):
    return r2k(freqs, DeltaI_mu(freqs, mu_amp))

# r-type distortion (first non-mu/non-y eigenmode -- this is only approximately correct for us to use here, but let's stick with it for now)
def DeltaI_r(freqs, r_amp=1.e-6): #freqs in Hz, r_amp dimensionless, DeltaI_r in W/m^2/Hz/sr
    X = hplanck*freqs/(kboltz*TCMB)
    # first r-distortion eigenmode from Jens (Fig. 4 of 1306.5751)
    rfile = np.loadtxt('templates/PCA_mode_1.dat')
    Xr = hplanck*rfile[:,0]*1e9/(kboltz*TCMB) #convert from GHz to Hz
    DeltaIr = rfile[:,1]*1e-18*r_amp #conver to W/m^2/Hz/sr
    # linearly interpolate (set to zero above the highest frequency in Jens's file (his lowest frequency is 30 GHz, so things are OK on that end--put in a crazy value so we catch it if needed))
    return np.interp(X, Xr, DeltaIr, left=-1e6, right=0.0)

def kDeltaI_r(freqs, r_amp=1.e-6):
    return r2k(freqs, DeltaI_r(freqs, r_amp))

# relativistic tSZ distortion ("beyond y") -- see Hill+2015
#   N.B. although this signal in principle requires an infinite number of parameters to be fully specified, in practice PIXIE will only be sensitive to (at best) one, which is
#      the mean tau-weighted ICM electron temperature (kT_moments[0], below).  We'll hold all the others fixed, or perhaps just put reasonable priors on them and marginalize over them.
def DeltaI_reltSZ(freqs, y_tot=1.77e-6, y_IGMplusreion=1.87e-7, kTmom1=0.208, kTmom2=0.299, kTmom3=0.892, kTmom4=4.02): #freqs in Hz, y parameters dimensionless, kT_moments in keV^n, DeltaI_reltSZ in W/m^2/Hz/sr; code uses up to kT_moments[3] (Eq. 8 of Hill+2015 with n=4)
    # immediately convert from ytot to tau_ICM
    tau_ICM = (y_tot - y_IGMplusreion)/kTmom1 * m_elec
    # convert to Jens's moment definitions -- this formalism follows Chluba+2013 (MNRAS, 430, 3054)
    w1 = (kTmom2)/(kTmom1)**2 - 1.0
    w2 = (kTmom3)/(kTmom1)**3 - 3.0*(kTmom2)/(kTmom1)**2 + 2.0
    w3 = (kTmom4)/(kTmom1)**4 - 4.0*(kTmom3)/(kTmom1)**3 + 6.0*(kTmom2)/(kTmom1)**2 - 3.0
    #w4 not included for now; stop at w3, which corresponds to <(kT)^4> order
    #w4 = (kTmom5)/(kTmom1)**5 - 5.0*(kTmom4)/(kTmom1)**4 + 10.0*(kTmom3)/(kTmom1)**3 - 10.0*(kTmom2)/(kTmom1)**2 + 4.0
    X = hplanck*freqs/(kboltz*TCMB)
    Xtwid = X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid = X/np.sinh(0.5*X)
    # relativistic expressions from Nozawa+2006
    Y0=Xtwid-4.0
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+0.3666667*Xtwid**5.0+Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+4.7666667*Xtwid**3.0)+Stwid**4.0*(-8.8+3.11666667*Xtwid)
    Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-531.257142857*Xtwid**4.0+86.1357142857*Xtwid**5.0-6.09523809524*Xtwid**6.0+0.15238095238*Xtwid**7.0+Stwid**2.0*(-709.8+2850.6*Xtwid-2921.91428571*Xtwid**2.0+1119.76428571*Xtwid**3.0-173.714285714*Xtwid**4.0+9.14285714286*Xtwid**5.0)+Stwid**4.0*(-531.257142857+732.153571429*Xtwid-274.285714286*Xtwid**2.0+29.2571428571*Xtwid**3.0)+Stwid**6.0*(-25.9047619048+9.44761904762*Xtwid)
    gfuncrel=Y0+Y1*(kTmom1/m_elec)+Y2*(kTmom1/m_elec)**2.0+Y3*(kTmom1/m_elec)**3.0 #third-order
    ddgfuncrel=2.0*Y1+6.0*Y2*(kTmom1/m_elec)+24.0*Y3*(kTmom1/m_elec)**2.0
    dddgfuncrel=6.0*Y2+24.0*Y3*(kTmom1/m_elec)
    ddddgfuncrel=24.0*Y3
    return X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * (y_IGMplusreion * Y0 + tau_ICM * (kTmom1/m_elec) * gfuncrel + tau_ICM/2.0 * ddgfuncrel * (kTmom1/m_elec)**2.0 * w1 + tau_ICM/6.0 * dddgfuncrel * (kTmom1/m_elec)**3.0 * w2 + tau_ICM/24.0 * ddddgfuncrel * (kTmom1/m_elec)**4.0 * w3)

def kDeltaI_reltSZ(freqs, y_tot=1.77e-6, y_IGMplusreion=1.87e-7, kTmom1=0.208, kTmom2=0.299, kTmom3=0.892, kTmom4=4.02):
    return r2k(freqs, DeltaI_reltSZ(freqs, y_tot, y_IGMplusreion, kTmom1, kTmom2, kTmom3, kTmom4))

# 2-parameter version of rel. tSZ -- only allow y_tot and kTmom1 to vary
def DeltaI_reltSZ_2param(freqs, y_tot=1.77e-6, kTmom1=0.208): #freqs in Hz, y parameters dimensionless, kT_moments in keV^n, DeltaI_reltSZ in W/m^2/Hz/sr; code uses up to kT_moments[3] (Eq. 8 of Hill+2015 with n=4)
    # parameters held fixed
    y_IGMplusreion=1.87e-7
    kTmom2=0.299
    kTmom3=0.892
    kTmom4=4.02
    # immediately convert from ytot to tau_ICM
    tau_ICM = (y_tot - y_IGMplusreion)/kTmom1 * m_elec
    # convert to Jens's moment definitions -- this formalism follows Chluba+2013 (MNRAS, 430, 3054)
    w1 = (kTmom2)/(kTmom1)**2 - 1.0
    w2 = (kTmom3)/(kTmom1)**3 - 3.0*(kTmom2)/(kTmom1)**2 + 2.0
    w3 = (kTmom4)/(kTmom1)**4 - 4.0*(kTmom3)/(kTmom1)**3 + 6.0*(kTmom2)/(kTmom1)**2 - 3.0
    #w4 not included for now; stop at w3, which corresponds to <(kT)^4> order
    #w4 = (kTmom5)/(kTmom1)**5 - 5.0*(kTmom4)/(kTmom1)**4 + 10.0*(kTmom3)/(kTmom1)**3 - 10.0*(kTmom2)/(kTmom1)**2 + 4.0
    X = hplanck*freqs/(kboltz*TCMB)
    Xtwid = X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid = X/np.sinh(0.5*X)
    # relativistic expressions from Nozawa+2006
    Y0=Xtwid-4.0
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+0.3666667*Xtwid**5.0+Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+4.7666667*Xtwid**3.0)+Stwid**4.0*(-8.8+3.11666667*Xtwid)
    Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-531.257142857*Xtwid**4.0+86.1357142857*Xtwid**5.0-6.09523809524*Xtwid**6.0+0.15238095238*Xtwid**7.0+Stwid**2.0*(-709.8+2850.6*Xtwid-2921.91428571*Xtwid**2.0+1119.76428571*Xtwid**3.0-173.714285714*Xtwid**4.0+9.14285714286*Xtwid**5.0)+Stwid**4.0*(-531.257142857+732.153571429*Xtwid-274.285714286*Xtwid**2.0+29.2571428571*Xtwid**3.0)+Stwid**6.0*(-25.9047619048+9.44761904762*Xtwid)
    gfuncrel=Y0+Y1*(kTmom1/m_elec)+Y2*(kTmom1/m_elec)**2.0+Y3*(kTmom1/m_elec)**3.0 #third-order
    ddgfuncrel=2.0*Y1+6.0*Y2*(kTmom1/m_elec)+24.0*Y3*(kTmom1/m_elec)**2.0
    dddgfuncrel=6.0*Y2+24.0*Y3*(kTmom1/m_elec)
    ddddgfuncrel=24.0*Y3
    return X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * (y_IGMplusreion * Y0 + tau_ICM * (kTmom1/m_elec) * gfuncrel + tau_ICM/2.0 * ddgfuncrel * (kTmom1/m_elec)**2.0 * w1 + tau_ICM/6.0 * dddgfuncrel * (kTmom1/m_elec)**3.0 * w2 + tau_ICM/24.0 * ddddgfuncrel * (kTmom1/m_elec)**4.0 * w3)

def kDeltaI_reltSZ_2param(freqs, y_tot=1.77e-6, kTmom1=0.208):
    return r2k(freqs, DeltaI_reltSZ_2param(freqs, y_tot, kTmom1))


# recombination lines from template
# file is in GHz and W / m^2 Hz sr
def recombination(freqs, scale=1.0):
    rdata = np.loadtxt('templates/recombination/total_spectrum_f.dat')
    fs = rdata[:,0] * 1e9
    recomb = rdata[:,1]
    template = interpolate.interp1d(np.log10(fs), np.log10(recomb), fill_value=np.log10(1e-30), bounds_error=False)
    return scale * 10.0**template(np.log10(freqs))

def krecombination(freqs, scale=1.0):
    return r2k(freqs, recombination(freqs, scale))

def pixie_sensitivity(freqs, fsky=0.7):
    sdata = np.loadtxt('Sensitivities.dat')
    fs = sdata[:,0] * 1e9
    sens = sdata[:,1]
    template = np.interp(np.log10(freqs), np.log10(fs), np.log10(sens), left=-23.3, right=-21.)
    skysr = 4.*np.pi*(180./np.pi)**2*fsky
    return 10.**template / np.sqrt(skysr)

def kpixie_sensitivity(freqs, fsky=0.7):
    return r2k(freqs, pixie_sensitivity(freqs, fsky))
