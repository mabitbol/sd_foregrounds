import numpy as np
import emcee
import inspect

import spectral_distortions as sd
import foregrounds as fg
import fisher

class ForecasterMCMC:
    def __init__(fncs=None, duration=86.4, nwalkers=0, gf=1.e-3, threads=16):    
        if fncs is None:
            print "need funcs" 
            return
        fish = fisher.FisherEstimation(duration=duration, bandpass=False)
        self.freqs = fish.center_frequencies
        self.noise = fish.noise
        self.fncs = fncs
        self.setup_fncs()
        self.generate_data()
        self.setup_mcmc(nwalkers, gf, threads)

    def setup_fncs(self):
        self.args, self.p0, self.argvals = self.get_function_args()
        self.setup_priors()
        
    def get_function_args(self):
        targs = []
        tp0 = []
        for fnc in self.fncs:
            argsp = inspect.getargspec(fnc)
            targs = np.concatenate([targs, argsp[0][1:]])
            tp0 = np.concatenate([tp0, argsp[-1]])
        return targs, tp0, dict(zip(targs, tp0))
    
    def generate_data(self):
        y = self.measure()
        yerr = self.noise*np.random.randn(len(self.freqs))
        self.data = y + yerr
    
    def setup_mcmc(self, nwalkers, gf, threads):
        self.ndim = len(self.args)
        if nwalkers>0:
            self.nwalkers = nwalkers
        else:
            selfnwalkers = int(2**np.ceil(np.log2(6*ndim)))
        self.pos = [self.p0*(1. + gf*np.random.randn(ndim)) for i in range(self.nwalkers)]

        for k in range(len(pos)):
            lp = self.lnprior(self.pos[k])
            if not np.isfinite(lp):
                print "bad"
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob,\
                                             args=(self.freqs, self.data, self.noise), threads=threads)
        
    def lnprob(self, theta, x, y, yerr):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, x, y, yerr)

    def measure(self):
        #MEASURE!!!!
    
    def setup_priors(self):
        #priors!!!

    def lnlike(self, theta, x, y, yerr):
        mu_amp, y_tot, kT_yweight, DeltaT_amp = theta
        model = sd.DeltaI_mu(x, mu_amp)+sd.DeltaI_reltSZ_2param_yweight(x, y_tot, kT_yweight)+\
                sd.DeltaI_DeltaT(x, DeltaT_amp)
        return -0.5 * (np.sum((y-model)**2. * yerr**-2.))


    def lnprior(self, theta):
        mu_amp, y_tot, kT_yweight, DeltaT_amp, As, alps, w2s, EM, Ad, Bd, Td, Acib, Bcib, Tcib = theta
        if 0<As and alps<0 and 0<EM and 0<Ad and 0<Bd and 0<Td and 0<Acib and 0<Bcib and 0<Tcib:
            return 0.
        return -np.inf



