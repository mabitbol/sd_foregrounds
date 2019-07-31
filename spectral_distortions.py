import numpy as np

### See components for a better description of the signals. 

TCMB = 2.7255 #Kelvin
hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS
clight=299792458.0 #MKS
m_elec = 510.999 #keV!
jy = 1.e26

ndp = np.float128

def DeltaI_DeltaT(freqs, DeltaT_amp=1.e-4):
    X = hplanck*freqs/(kboltz*TCMB)
    return (DeltaT_amp * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * jy).astype(ndp)

def DeltaI_mu(freqs, mu_amp=2.e-8):
    X = hplanck*freqs/(kboltz*TCMB)
    return (mu_amp * (X / 2.1923 - 1.0)/X * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * jy).astype(ndp)

def DeltaI_reltSZ_2param_yweight(freqs, y_tot=2.e-6, kT_yweight=1.3):
    tau = y_tot/kT_yweight * m_elec
    X = hplanck*freqs/(kboltz*TCMB)
    Xtwid = X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid = X/np.sinh(0.5*X)
    Y0=Xtwid-4.0
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+0.3666667*Xtwid**5.0+Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+4.7666667*Xtwid**3.0)+Stwid**4.0*(-8.8+3.11666667*Xtwid)
    Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-531.257142857*Xtwid**4.0+86.1357142857*Xtwid**5.0-6.09523809524*Xtwid**6.0+0.15238095238*Xtwid**7.0+Stwid**2.0*(-709.8+2850.6*Xtwid-2921.91428571*Xtwid**2.0+1119.76428571*Xtwid**3.0-173.714285714*Xtwid**4.0+9.14285714286*Xtwid**5.0)+Stwid**4.0*(-531.257142857+732.153571429*Xtwid-274.285714286*Xtwid**2.0+29.2571428571*Xtwid**3.0)+Stwid**6.0*(-25.9047619048+9.44761904762*Xtwid)
    gfuncrel_only=Y1*(kT_yweight/m_elec)+Y2*(kT_yweight/m_elec)**2.0+Y3*(kT_yweight/m_elec)**3.0
    return (X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * (y_tot * Y0 + tau * (kT_yweight/m_elec) * gfuncrel_only) * jy).astype(ndp)

def DeltaI_y(freqs, y_tot=2.e-6):
    X = hplanck*freqs/(kboltz*TCMB)
    return ((y_tot * (X / np.tanh(X/2.0) - 4.0) * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0) * jy).astype(ndp)

def blackbody(nu, T=2.725):
    X = hplanck * nu / (kboltz * T)
    bbT = jy * 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(X) - 1.0))
    return bbT.astype(ndp)

def diff_blackbody(nu, deltaT=1.e-3):
    T = TCMB - TCMB * deltaT
    Xcmb = hplanck * nu / (kboltz * TCMB)
    X = hplanck * nu / (kboltz * T)
    bbTcmb = jy * 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(Xcmb) - 1.0))
    bbT = jy * 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(X) - 1.0))
    return (bbTcmb - bbT).astype(ndp)

def blackbody_H0(nu, H0=7.2e-11):
    T = TCMB - TCMB * H0 * 1. # 1yr
    Xcmb = hplanck * nu / (kboltz * TCMB)
    X = hplanck * nu / (kboltz * T)
    bbTcmb = jy * 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(Xcmb) - 1.0))
    bbT = jy * 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(X) - 1.0))
    return (bbTcmb - bbT).astype(ndp)

def recombination(freqs, scale=1.0):
    rdata = np.loadtxt('templates/CRR.dat')
    fs = rdata[:,0] * 1e9
    recomb = rdata[:,1]
    return scale * np.interp(freqs, fs, recomb)
