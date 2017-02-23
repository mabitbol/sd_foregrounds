import numpy as np
import emcee

import spectral_distortions as sd
import foregrounds as fg
import fisher

#import sys
#from emcee.utils import MPIPool

import multiprocessing as mp

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def lnlike(theta, x, y, yerr):
    mu_amp, y_tot, kT_yweight, DeltaT_amp = theta
    model = sd.DeltaI_mu(x, mu_amp)+sd.DeltaI_reltSZ_2param_yweight(x, y_tot, kT_yweight)+\
            sd.DeltaI_DeltaT(x, DeltaT_amp)
    return -0.5 * (np.sum((y-model)**2. * yerr**-2.))

def lnprior(theta):
    return 0.

def run():
    pool = mp.Pool()

    nargs = ['mu', 'y', 'kT', 'DeltaT']
    fish = fisher.FisherEstimation(duration=12., bandpass=False)
    fncs = [sd.DeltaI_mu, sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT]
    fish.set_signals(fncs)
    x = fish.center_frequencies
    noise = fish.noise / 1000.
    y = sd.DeltaI_mu(x)+sd.DeltaI_reltSZ_2param_yweight(x)+sd.DeltaI_DeltaT(x)
    yerr = noise*np.random.randn(len(x))
    y += yerr
    ndim, nwalkers = 4, 32
    pos = [fish.p0*(1. + 1.e-1*np.random.randn(ndim)) for i in range(nwalkers)] 

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, noise), pool=pool)
    sampler.run_mcmc(pos, 2000);
    pool.close()

    samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    p_mc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    for k in range(4):
        print nargs[k], (p_mc[k][1]+p_mc[k][2])/2.
    return

run()
