import numpy as np

import fisher
import spectral_distortions as sd
import foregrounds as fg

import emcee
import corner

def lnprob(theta, x, y, noise):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf, -np.inf
    return lnlike(theta, x, y, noise), lp

def lnlike(theta, x, y, noise):
    dt, ytot, Ad, Bd, Td, Acib, Bcib, Tcib, co = theta
    model = sd.DeltaI_DeltaT(x, dt)
    model += sd.DeltaI_y(x, ytot)
    model += fg.thermal_dust(x, Ad, Bd, Td)
    model += fg.cib(x, Acib, Bcib, Tcib)
    model += fg.co(x, co)
    return -0.5 * (np.sum((y-model)**2. * noise**-2.))

def lnprior(theta):
    dt, ytot, Ad, Bd, Td, Acib, Bcib, Tcib, co = theta
    if np.abs(dt) >= 1.:
        return -np.inf
    if np.abs(ytot) >= 1.:
        return -np.inf

    if Ad < 0 or Acib < 0:
        return -np.inf
    if co < 0: 
        return -np.inf
    if Bd < 0 or Bcib < 0: 
        return -np.inf
    if Td < 1 or Td > 100:
        return -np.inf
    if Tcib < 1 or Tcib > 100:
        return -np.inf
    return 0.

def run():
    fname = 'allfgin_fitHFonly'
    output_dir = make_output_dir(fname)

    nwalkers = 512
    nsamps = 10000000

    bx = 30. * (12. / 8760) # in months
    fmin = 82.5e9
    fsky = 1.
    p0 = {}

    sigs = [sd.DeltaI_DeltaT, sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_mu, \
            fg.thermal_dust, fg.cib, fg.co, fg.synch, fg.freefree, fg.spinning_dust]
    fish = fisher.FisherEstimation(duration=bx, fmin=fmin, fsky=fsky, priors=p0, bandpass=False, fncs=sigs)
    #fish.run_fisher_calculation()
    #fish.print_errors()

    x = fish.center_frequencies
    noise = fish.noise
    y = np.zeros(len(x))
    for sig in sigs:
        y += sig(x)
    yerr = noise*np.random.randn(len(x))
    #y += yerr

    # hack to grab initial values
    sigs = [sd.DeltaI_DeltaT, sd.DeltaI_y, fg.thermal_dust, fg.cib, fg.co]
    fish = fisher.FisherEstimation(duration=bx, fmin=fmin, fsky=fsky, priors=p0, bandpass=False, fncs=sigs)
    fish.run_fisher_calculation()
    np.savez(output_dir+'argvals', names=fish.args, p0=fish.p0, errors=fish.errors, x=x, y=y, noise=noise) 

    chain_fname = output_dir + 'chains.h5'
    ndim = len(fish.args)
    pos = [fish.p0*(1. + 1.e-2*np.random.randn(ndim)) for i in range(nwalkers)]
    backend = emcee.backends.HDFBackend(chain_fname)
    backend.reset(nwalkers, ndim) 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, noise), backend=backend)
    
    old_tau = np.inf
    for sample in sampler.sample(pos, iterations=nsamps, progress=False):
        if sampler.iteration % 100:
            continue
        tau = sampler.get_autocorr_time(tol=0)
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs( old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
    if not converged:
        print "Did not converge."
        np.savez(output_dir+'chains.npz', chains=sampler.chain)
    return 


def make_output_dir(addname):
    from datetime import datetime
    import os, errno
    from shutil import copyfile
    fmt='%Y-%m-%d-%H-%M'
    date = datetime.now().strftime(fmt)
    output_dir = 'outputs/'+date + '_' + addname
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    copyfile('sampler.py', output_dir+'/sampler.py') 
    return output_dir + '/'

run()