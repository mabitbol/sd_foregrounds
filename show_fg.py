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
from foregrounds import *

PIXIE_freq_min = 37.5e9 #central frequency of lowest channel (lower bound is 30 GHz)
PIXIE_freq_max = 6.0225e12 #central frequency of highest channel (chose this to get 400 total channels)
PIXIE_freqstep = 15.0e9

def pixie_frequencies(): # PIXIE frequency channels (all in Hz) -- see http://arxiv.org/abs/1105.2044
    PIXIE_freqs = np.arange(PIXIE_freq_min, PIXIE_freq_max + PIXIE_freqstep, PIXIE_freqstep)
    return PIXIE_freqs

def pixie_noise(PIXIE_freqs):
    PIXIE_Nfreqs = len(PIXIE_freqs)
    PIXIE_noise = 5.0e-26*np.ones(PIXIE_Nfreqs) #http://arxiv.org/abs/1105.2044 ; taken to be uncorrelated between channels; units W/m^2/Hz/sr 
    return PIXIE_noise

def spectral_distortions(PIXIE_freqs, DeltaT_fid=1.2e-4, y_fid=1.77e-6, mu_fid=2.0e-8, r_fid=1.0e-6,\
                         tau_ICM_fid=3.89e-3, kT_moments_fid=[0.208, 0.299, 0.892, 4.02]):
    # compute each component
    PIXIE_DeltaI_DeltaT = DeltaI_DeltaT(PIXIE_freqs, DeltaT_fid)
    PIXIE_DeltaI_y = DeltaI_y(PIXIE_freqs, y_fid)
    PIXIE_DeltaI_mu = DeltaI_mu(PIXIE_freqs, mu_fid)
    PIXIE_DeltaI_r = DeltaI_r(PIXIE_freqs, r_fid)
    PIXIE_DeltaI_reltSZ = DeltaI_reltSZ(PIXIE_freqs, tau_ICM_fid, kT_moments_fid)
    # total
    #PIXIE_DeltaI_tot = PIXIE_DeltaI_DeltaT + PIXIE_DeltaI_y + PIXIE_DeltaI_mu + PIXIE_DeltaI_r + PIXIE_DeltaI_reltSZ
    return PIXIE_DeltaI_DeltaT, PIXIE_DeltaI_y, PIXIE_DeltaI_mu, PIXIE_DeltaI_r, PIXIE_DeltaI_reltSZ

def foregrounds(PIXIE_freqs):
    dust = krj_to_radiance(PIXIE_freqs, thermal_dust(PIXIE_freqs))
    synch = krj_to_radiance(PIXIE_freqs, synchrotron(PIXIE_freqs))
    brem = krj_to_radiance(PIXIE_freqs, freefree(PIXIE_freqs))
    spinning = krj_to_radiance(PIXIE_freqs, ame(PIXIE_freqs))
    #foregrounds = dust+synch+brem+spinning+sunyaev
    return dust, synch, brem, spinning 

def plot_foregrounds():
    PIXIE_freqs = pixie_frequencies()
    dust, synch, brem, spinning = foregrounds(PIXIE_freqs)
    plt.clf()
    plt.figure()
    plt.title('foregrounds', fontsize=20)
    #plt.ylabel('Jy/sr')
    plt.ylabel(r'$\langle \Delta I_{\nu} \rangle \times 10^{23} \,\, [{\rm W / m^2 / Hz/ sr}]$', fontsize=20)    
    plt.xlabel(r'$\nu \, [{\rm GHz}]$', fontsize=20)
    plt.xlim( PIXIE_freq_min/1e9, PIXIE_freq_max/1e9)
    #plt.ylim(10**-1, 10**10)
    plt.ylim(1.0e-4,1.0e5)
    a = plt.gca()
    a.set_xticklabels(a.get_xticks(), fontProperties)
    a.set_yticklabels(a.get_yticks(), fontProperties)
    for label in a.get_xticklabels() + a.get_yticklabels():
        label.set_fontsize(18)
    plt.loglog(PIXIE_freqs/1.e9, 1.0e23*dust, label='dust')
    plt.loglog(PIXIE_freqs/1.e9, 1.0e23*synch, label='synch')
    plt.loglog(PIXIE_freqs/1.e9, 1.0e23*brem, label='free free')
    plt.loglog(PIXIE_freqs/1.e9, 1.0e23*spinning, label='ame')
    plt.grid()
    plt.legend()
    plt.savefig('allfg_v2.pdf')

def plot_distortions():
    PIXIE_freqs = pixie_frequencies()
    PIXIE_DeltaI_DeltaT, PIXIE_DeltaI_y, PIXIE_DeltaI_mu, PIXIE_DeltaI_r, PIXIE_DeltaI_reltSZ = spectral_distortions(PIXIE_freqs)
    PIXIE_DeltaI_tot = PIXIE_DeltaI_DeltaT + PIXIE_DeltaI_y + PIXIE_DeltaI_mu + PIXIE_DeltaI_r + PIXIE_DeltaI_reltSZ
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

def plot_all():
    PIXIE_freqs = pixie_frequencies()
    PIXIE_DeltaI_DeltaT, PIXIE_DeltaI_y, PIXIE_DeltaI_mu, PIXIE_DeltaI_r, PIXIE_DeltaI_reltSZ = spectral_distortions(PIXIE_freqs)
    PIXIE_DeltaI_tot = PIXIE_DeltaI_DeltaT + PIXIE_DeltaI_y + PIXIE_DeltaI_mu + PIXIE_DeltaI_r + PIXIE_DeltaI_reltSZ
    dust, synch, brem, spinning = foregrounds(PIXIE_freqs)
    plt.clf()
    plt.figure()
    plt.xlabel(r'$\nu \, [{\rm GHz}]$', fontsize=20)
    plt.ylabel(r'$|\langle \Delta I_{\nu} \rangle| \times 10^{23} \,\, [{\rm W / m^2 / Hz/ sr}]$', fontsize=20)
    plt.xlim( PIXIE_freq_min/1e9, PIXIE_freq_max/1e9)
    plt.ylim(1.0e-4,1.0e5)
    a = plt.gca()
    a.set_xticklabels(a.get_xticks(), fontProperties)
    a.set_yticklabels(a.get_yticks(), fontProperties)
    for label in a.get_xticklabels() + a.get_yticklabels():
        label.set_fontsize(18)
    DeltaT_pts, = plt.loglog(PIXIE_freqs/1e9, np.absolute(PIXIE_DeltaI_DeltaT*1e23), 'r', lw=2.5)
    y_pts, = plt.loglog(PIXIE_freqs/1e9, np.absolute(PIXIE_DeltaI_y*1e23), 'b', lw=2.5)
    mu_pts, = plt.loglog(PIXIE_freqs/1e9, np.absolute(PIXIE_DeltaI_mu*1e25), 'g', lw=2.5)
    r_pts, = plt.loglog(PIXIE_freqs/1e9, np.absolute(PIXIE_DeltaI_r*1e25), 'orange', lw=2.5)
    reltSZ_pts, = plt.loglog(PIXIE_freqs/1e9, np.absolute(PIXIE_DeltaI_reltSZ*1e24), 'c', lw=2.5)
    plt.loglog(PIXIE_freqs/1.e9, 1.0e23*dust, label='dust')
    plt.loglog(PIXIE_freqs/1.e9, 1.0e23*synch, label='synch')
    plt.loglog(PIXIE_freqs/1.e9, 1.0e23*brem, label='free free')
    plt.loglog(PIXIE_freqs/1.e9, 1.0e23*spinning, label='ame')
    plt.grid()
    #plt.figlegend( (DeltaT_pts, y_pts, reltSZ_pts, mu_pts, r_pts), (r'$\Delta_T$',r'non-relativistic tSZ (pure $y$)',r'relativistic tSZ corrections $\times$ 10',r'$\mu \times 100$',r'$r$-type $\times$ 100'), loc='upper right', bbox_to_anchor=(0.9, 0.9), fontsize=14, ncol=2)
    plt.savefig('all_comps.pdf')


plot_foregrounds()
plot_all()
