import numpy as np
import math, os
import glob
from scipy import interpolate
from scipy import integrate
from scipy import special
import matplotlib
matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
    'weight' : 'normal', 'size' : 20}
import matplotlib.pyplot as plt
from components import *

"""
This code specifies amplitudes for the components and computes the full signal.
"""

# constants (MKS units)
TCMB = 2.726 #Kelvin
hplanck = 6.626068e-34 #MKS
kboltz = 1.3806503e-23 #MKS
clight = 299792458.0 #MKS

# amplitudes -- probably want to put these into a vector (i.e., array)
#   the vector should also include any parameters describing the SEDs that we want to consider variations of (e.g., dust spectral index).
#   this parameter vector will form the basis of the Fisher calculation later.
DeltaT_fid = 1.2e-4 #from 1306.5751 Section 4
y_fid = 1.77e-6 #see Hill+2015
mu_fid = 2.0e-8 #see Chluba, Khatri, & Sunyaev 2012 (MNRAS 425, 1129)
r_fid = 1.0e-6 #see 1306.5751, where this is called "mu_1"
tau_ICM_fid = 3.89e-3 #Hill+2015 pg 4
kT_moments_fid = [0.208, 0.299, 0.892, 4.02] #Hill+2015 pg 4 (units are keV, keV^2, keV^3, keV^4)

# PIXIE frequency channels (all in Hz) -- see http://arxiv.org/abs/1105.2044
PIXIE_freq_min = 37.5e9 #central frequency of lowest channel (lower bound is 30 GHz)
PIXIE_freq_max = 6.0225e12 #central frequency of highest channel (chose this to get 400 total channels)
PIXIE_freqstep = 15.0e9
PIXIE_freqs = np.arange(PIXIE_freq_min, PIXIE_freq_max + PIXIE_freqstep, PIXIE_freqstep)
PIXIE_Nfreqs = len(PIXIE_freqs)
PIXIE_noise = 5.0e-26*np.ones(PIXIE_Nfreqs) #http://arxiv.org/abs/1105.2044 ; taken to be uncorrelated between channels; units W/m^2/Hz/sr 

# compute each component
PIXIE_DeltaI_DeltaT = DeltaI_DeltaT(PIXIE_freqs, DeltaT_fid)
PIXIE_DeltaI_y = DeltaI_y(PIXIE_freqs, y_fid)
PIXIE_DeltaI_mu = DeltaI_mu(PIXIE_freqs, mu_fid)
PIXIE_DeltaI_r = DeltaI_r(PIXIE_freqs, r_fid)
PIXIE_DeltaI_reltSZ = DeltaI_reltSZ(PIXIE_freqs, tau_ICM_fid, kT_moments_fid)
# total
PIXIE_DeltaI_tot = PIXIE_DeltaI_DeltaT + PIXIE_DeltaI_y + PIXIE_DeltaI_mu + PIXIE_DeltaI_r + PIXIE_DeltaI_reltSZ

dust = thermal_dust(PIXIE_freqs)
synch = synchrotron(PIXIE_freqs)
brem = freefree(PIXIE_freqs)
spinning = ame(PIXIE_freqs)
sunyaev = sz(PIXIE_freqs)
foregrounds = dust+synch+brem+spinning+sunyaev

plt.figure()
plt.loglog(PIXIE_freqs/1.e9, dust, label='dust')
plt.xlabel('GHz')
plt.ylabel('uKrj')
plt.grid()
plt.savefig('dust')

plt.figure()
plt.loglog(PIXIE_freqs/1.e9, synch, label='synch')
plt.xlabel('GHz')
plt.ylabel('uKrj')
plt.grid()
plt.savefig('synch')

plt.figure()
plt.loglog(PIXIE_freqs/1.e9, brem, label='free free')
plt.xlabel('GHz')
plt.ylabel('uKrj')
plt.grid()
plt.savefig('brem')

plt.figure()
plt.loglog(PIXIE_freqs/1.e9, spinning, label='ame')
plt.xlabel('GHz')
plt.ylabel('uKrj')
plt.grid()
plt.savefig('ame')

plt.figure()
plt.semilogx(PIXIE_freqs/1.e9, sunyaev, label='sz')
plt.xlabel('GHz')
plt.ylabel('uKrj')
plt.grid()
plt.savefig('sz')

plt.figure()
plt.semilogx(PIXIE_freqs/1.e9, foregrounds, label='total')
plt.grid()
plt.legend()
plt.xlabel(r'$\nu \, [{\rm GHz}]$', fontsize=20)
plt.savefig('foregroundtest')


# plot
if False:
    plt.clf()
    plt.figure()
    plt.xlabel(r'$\nu \, [{\rm GHz}]$', fontsize=20)
    plt.ylabel(r'$\langle \Delta I_{\nu} \rangle \times 10^{23} \,\, [{\rm W / m^2 / Hz/ sr}]$', fontsize=20)
    plt.xlim( PIXIE_freq_min/1e9, PIXIE_freq_max/1e9)
    plt.ylim( -3.0, 5.0 )
    a = plt.gca()
    a.set_xticklabels(a.get_xticks(), fontProperties)
    a.set_yticklabels(a.get_yticks(), fontProperties)
    for label in a.get_xticklabels() + a.get_yticklabels():
        label.set_fontsize(18)
    plt.axhline(y=0.0, xmin=0.0, xmax=1.0, color='k', linewidth=1.5)
    DeltaT_pts, = plt.semilogx(PIXIE_freqs/1e9, PIXIE_DeltaI_DeltaT*1e23, 'r', lw=2.5)
    y_pts, = plt.semilogx(PIXIE_freqs/1e9, PIXIE_DeltaI_y*1e23, 'b', lw=2.5)
    mu_pts, = plt.semilogx(PIXIE_freqs/1e9, PIXIE_DeltaI_mu*1e25, 'g', lw=2.5)
    r_pts, = plt.semilogx(PIXIE_freqs/1e9, PIXIE_DeltaI_r*1e25, 'orange', lw=2.5)
    reltSZ_pts, = plt.semilogx(PIXIE_freqs/1e9, PIXIE_DeltaI_reltSZ*1e24, 'c', lw=2.5)
    tot_pts, = plt.semilogx(PIXIE_freqs/1e9, PIXIE_DeltaI_tot*1e23, 'k', lw=1.5)
    plt.grid()
    plt.figlegend( (DeltaT_pts, y_pts, reltSZ_pts, mu_pts, r_pts, tot_pts), (r'$\Delta_T$',r'non-relativistic tSZ (pure $y$)',r'relativistic tSZ corrections $\times$ 10',r'$\mu \times 100$',r'$r$-type $\times$ 100',r'total'), loc='upper right', bbox_to_anchor=(0.9, 0.9), fontsize=14, ncol=2)
    plt.savefig('PIXIE_DeltaT_y_mu_r_reltSZ_tot.pdf')
