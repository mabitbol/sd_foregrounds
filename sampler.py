import numpy as np
import time

import fisher
import spectral_distortions as sd
import foregrounds as fg

import emcee
import corner

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


def lnlike(theta, x, y, yerr):
    y, kt, dt, ad, bd, td, As, alps, w2s = theta
    model = sd.DeltaI_reltSZ_2param_yweight(x, y, kt)
    model += sd.DeltaI_DeltaT(x, dt)
    model += fg.thermal_dust_rad(x, ad, bd, td)
    model += fg.jens_synch_rad(x, As, alps, w2s)
    return -0.5 * (np.sum((y-model)**2. * yerr**-2.))

def lnprior(theta):
    y, kt, dt, ad, bd, td, As, alps, w2s = theta
    if ad < 0:
        return -np.inf
    if As < 0:
        return -np.inf
    if np.abs(bd) > 10:
        return -np.inf
    if td <= 0 or td > 100:
        return -np.inf
    if kt < 0 or kt>100:
        return -np.inf
    if np.abs(alps) > 10:
        return -np.inf
    if np.abs(w2s) > 10:
        return -np.inf
    return 0.

fmin = 82.5e9
p0 = {}
fsky = 1.
sigs = [sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT, \
        fg.thermal_dust_rad, fg.jens_synch_rad]

bx = 30. * (12. / 8760) # months
fish = fisher.FisherEstimation(duration=bx, fmin=fmin, fsky=fsky, priors=p0, bandpass=False)
fish.set_signals(sigs)
fish.run_fisher_calculation()
fish.print_errors()

x = fish.center_frequencies
noise = fish.noise
y = np.zeros(len(x))
for sig in sigs:
    y += sig(x)
yerr = noise*np.random.randn(len(x))
y += yerr

start_time = time.time()
ndim, nwalkers, nsteps = len(fish.args), 256, 1000
pos = [fish.p0*(1. + 1.e-1*np.random.randn(ndim)) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, noise))
sampler.run_mcmc(pos, nsteps);
print (time.time() - start_time)/60., 'mins'

np.savez('wowfiledotcom', chain=sampler.chain, names=fish.args)
