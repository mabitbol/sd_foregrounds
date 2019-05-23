import inspect
import numpy as np
from scipy import interpolate

import spectral_distortions as sd
import foregrounds as fg

from mpmath import mp, matrix

ndp = np.float128
mp.prec = 500

class FisherEstimation:
    def __init__(self, mults=[1., 1., 1.], dfms=[1., 1., 1.], priors={}, fncs=None, hfmax=6.e3):
        self.priors = priors

        self.setup_channels_noise(mults, dfms, hfmax)
        self.set_signals(fncs)
        return

    def setup_channels_noise(self, mults, dfms, hfmax):
        oneyr = 365.25 * 24. * 3600. # 1yr in seconds
        ghz = 1.e9
        jy = 1.e26

        lf_data = np.loadtxt('templates/write_apc_fts_noise_low.txt')
        mf_data = np.loadtxt('templates/write_apc_fts_noise_mid.txt')
        hf_data = np.loadtxt('templates/write_apc_fts_noise_high.txt')
    
        lf_nu = lf_data[:, 0] 
        mf_nu = mf_data[:, 0]
        hf_nu = hf_data[:, 0] 
        hfcut = hf_nu < hfmax

        lf_noise = lf_data[:, 1] * mults[0]
        mf_noise = mf_data[:, 1] * mults[1]
        hf_noise = hf_data[:, 1] * mults[2]
    
        hf_nu = hf_nu[hfcut]
        hf_noise = hf_noise[hfcut]

        lf_nu, lf_noise = self.interpolate_noise(lf_nu, lf_noise, dfms[0])
        mf_nu, mf_noise = self.interpolate_noise(mf_nu, mf_noise, dfms[1])
        hf_nu, hf_noise = self.interpolate_noise(hf_nu, hf_noise, dfms[2])

        self.frequencies = np.concatenate((lf_nu, mf_nu, hf_nu)) * ghz
        self.noise = np.concatenate((lf_noise, mf_noise, hf_noise)) / np.sqrt(oneyr) * jy
        return 

    def interpolate_noise(self, nus, noise, dfm):
        nustep = nus[1] - nus[0]
        numin = nus.min() - nustep/2.
        numax = nus.max() + nustep/2.

        newstep = nustep * dfm
        N = int(np.ceil( (numax-numin)/newstep ))
        newnus = numin + newstep/2. + np.arange(N) * newstep
        newnoise = np.interp(newnus, nus, noise) * (nustep / newstep)
        return newnus, newnoise

    def run_fisher_calculation(self):
        N = len(self.args)
        F = self.calculate_fisher_matrix()
        self.F = F
        for k in self.priors.keys():
            if k in self.args and self.priors[k] > 0:
                kindex = np.where(self.args == k)[0][0]
                F[kindex, kindex] += 1. / (self.priors[k] * self.argvals[k])**2
        print(mp)
        self.cov = matrix(F)**-1        
        self.get_errors()
        return

    def get_errors(self):
        self.errors = {}
        for k, arg in enumerate(self.args):
            self.errors[arg] = np.sqrt(self.cov[k,k])
        return

    def print_errors(self, args=None):
        if not args:
            args = self.args
        for arg in args:
            print arg, np.abs(self.argvals[arg]) / self.errors[arg]

    def set_signals(self, fncs=None):
        if fncs is None:
            fncs = [sd.DeltaI_DeltaT, sd.DeltaI_mu, sd.DeltaI_reltSZ_2param_yweight,
                    fg.thermal_dust, fg.cib, fg.freefree, fg.synch,
                    fg.spinning_dust, fg.co]
        self.signals = fncs
        self.args, self.p0, self.argvals = self.get_function_args()
        return

    def get_function_args(self):
        targs = []
        tp0 = []
        for fnc in self.signals:
            argsp = inspect.getargspec(fnc)
            args = argsp[0][1:]
            p0 = argsp[-1]
            targs = np.concatenate([targs, args])
            tp0 = np.concatenate([tp0, p0])
        return targs, tp0, dict(zip(targs, tp0))

    def calculate_fisher_matrix(self):
        N = len(self.p0)
        F = np.zeros([N, N], dtype=ndp)
        for i in range(N):
            dfdpi = self.signal_derivative(self.args[i], self.p0[i])
            dfdpi /= self.noise
            for j in range(N):
                dfdpj = self.signal_derivative(self.args[j], self.p0[j])
                dfdpj /= self.noise
                F[i, j] = np.dot(dfdpi, dfdpj)
        return F

    def signal_derivative(self, x, x0):
        h = 1.e-4
        zp = 1. + h
        deriv = (self.measure_signal(**{x: x0 * zp}) - self.measure_signal(**{x: x0})) / (h * x0)
        return deriv

    def measure_signal(self, **kwarg):
        N = len(self.frequencies)
        model = np.zeros(N, dtype=ndp)
        for fnc in self.signals:
            argsp = inspect.getargspec(fnc)
            args = argsp[0][1:]
            if len(kwarg) and kwarg.keys()[0] in args:
                model += fnc(self.frequencies, **kwarg)
        return model

