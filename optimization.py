import numpy as np
import emcee
import spectral_distortions as sd
import foregrounds as fg
import fisher


def lnlike(theta):
    fs, ns = theta[:200], theta[200:]
    
    fish = fisher.FisherEstimation(bandpass=False, priors={})
    fish.set_signals([sd.DeltaI_mu, sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT, \
                      fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad, fg.jens_synch_rad])
    fish.center_frequencies = fs
    fish.noise = ns
    fish.run_fisher_calculation()
    if np.isnan(fish.errors['mu_amp']):
        return -np.inf
    return -(fish.errors['mu_amp'] / 2.e-8  - 1./3.)**2


def lnprior(theta):
    fs, ns = theta[:200], theta[200:]
    
    if np.any(fs < 1e9):
        return np.inf
    if np.any(fs > 3.01e12):
        return np.inf
    if np.any(np.diff(fs) < 1e9):
        return np.inf
    if np.any(ns < 1.e-2):
        return np.inf
    if np.any(ns > 1.e2):
        return np.inf
    return 0.


def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

fish = fisher.FisherEstimation(bandpass=False, priors={})
fish.set_signals([sd.DeltaI_mu, sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT, \
                  fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad, fg.jens_synch_rad])
fs0 = fish.center_frequencies
noise0 = fish.noise / 10.
p0 = np.concatenate([fs0, noise0])
ndim, nwalkers = 400, 1024
pos = [p0 + 1.e-3*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(pos, 64);
np.savez('optimize1', sampler)
