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
    y_tot, dt = theta
    model = sd.DeltaI_DeltaT(x, dt)
    model += sd.DeltaI_y(x, y_tot)
    return -0.5 * (np.sum((y-model)**2. * yerr**-2.))

def lnprior(theta):
    y_tot, dt = theta
    if np.abs(y_tot) >= 1.:
        return -np.inf
    if np.abs(dt) >= 1.:
        return -np.inf
    return 0.

def run():
    fname = 'backendfuckup'
    nwalkers = 128
    nsamps = 1000
    check_convergence = True

    bx = 30. * (12. / 8760) # months
    fmin = 82.5e9
    fsky = 1.
    p0 = {}
    sigs = [sd.DeltaI_DeltaT, sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_mu]
    fish = fisher.FisherEstimation(duration=bx, fmin=fmin, fsky=fsky, priors=p0, bandpass=False, fncs=sigs)
    fish.run_fisher_calculation()
    fish.print_errors()

    x = fish.center_frequencies
    noise = fish.noise
    y = np.zeros(len(x))
    for sig in sigs:
        y += sig(x)
    yerr = noise*np.random.randn(len(x))
    #y += yerr

    # hack to grab initial values
    sigs = [sd.DeltaI_DeltaT, sd.DeltaI_y]
    fish = fisher.FisherEstimation(duration=bx, fmin=fmin, fsky=fsky, priors=p0, bandpass=False, fncs=sigs)
    fish.run_fisher_calculation()
    ndim = len(fish.args)
    pos = [fish.p0*(1. + 1.e-2*np.random.randn(ndim)) for i in range(nwalkers)]

    output_dir = make_output_dir(fname)
    start_time = time.time()
    if check_convergence:
        chain_fname = output_dir + 'chains.h5'
        backend = emcee.backends.HDFBackend(chain_fname)
        backend.reset(nwalkers, ndim) 
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, noise), backend=backend)
        
        old_tau = np.inf
        for sample in sampler.sample(pos, iterations=nsamps):
            if sampler.iteration % 100:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs( old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau

        if converged:
            tau = sampler.get_autocorr_time()
            burnin = int(2*np.max(tau))
            thin = int(0.5*np.min(tau))
            samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
            log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
            log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

            print("burn-in: {0}".format(burnin))
            print("thin: {0}".format(thin))
            print("flat chain shape: {0}".format(samples.shape))
            print("flat log prob shape: {0}".format(log_prob_samples.shape))
            print("flat log prior shape: {0}".format(log_prior_samples.shape))
        else:
            print "did not converge"
    else: 
        chain_fname = output_dir + 'chains.npz'
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, noise))
        sampler.run_mcmc(pos, nsamps);
        np.savez('chain_fname', chain=sampler.chain)

    print (time.time() - start_time)/60., 'mins'
    np.savez(output_dir+'argvals', names=fish.args, p0=fish.p0, errors=fish.errors) 
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
