import numpy as np
import foregrounds as fg
import components as sd
import inspect
from foregrounds import radiance_to_krj as r2k


class FisherEstimation:
    def __init__(self, fmin=8.e9, fmax=3.e12, fstep=15.e9, duration=86.4, bandpass=True, fsky=0.7, mult=1., prior=0.1):
        self.fmin = fmin
        self.fmax = fmax
        self.fstep = fstep
        self.duration = duration
        self.bandpass = bandpass
        self.fsky = fsky
        self.mult = mult
        self.prior = prior

        self.setup()
        return

    def setup(self):
        self.set_signals()
        self.set_frequencies()
        self.noise = self.kpixie_sensitivity()
        return

    def run_fisher_calculation(self):
        N = len(self.args)
        F = self.calculate_fisher_matrix()
        if self.prior > 0:
            alps_index = np.where(self.args == 'alps')[0][0]
            F[alps_index, alps_index] += 1. / (self.prior * self.p0[alps_index]) ** 2
        normF = np.zeros([N, N])
        for k in range(N):
            normF[k, k] = 1. / F[k, k]
        cov = (np.mat(normF) * np.mat(F)).I * np.mat(normF)
        return F, cov

    def set_signals(self, fncs=None):
        if fncs is None:
            fncs = [sd.kDeltaI_mu, sd.kDeltaI_reltSZ_2param_yweight, sd.kDeltaI_DeltaT,
                    fg.jens_freefree1p, fg.jens_synch, fg.cib, fg.spinning_dust, fg.co]
        self.signals = fncs
        self.args, self.p0 = self.get_function_args()
        return

    def set_frequencies(self):
        if self.bandpass:
            self.band_frequencies, self.center_frequencies, self.binstep = self.band_averaging_frequencies()
        else:
            self.center_frequencies = np.arange(self.fmin, self.fmax + self.fstep, self.fstep)
        return

    def band_averaging_frequencies(self):
        freqs = np.arange(self.fmin, self.fmax + self.fstep, 1.e9)
        binstep = int(self.fstep / 1.e9)
        freqs = freqs[:(len(freqs) / binstep) * binstep]
        centerfreqs = freqs.reshape((len(freqs) / binstep, binstep)).mean(axis=1)
        return freqs, centerfreqs, binstep

    def pixie_sensitivity(self):
        sdata = np.loadtxt('templates/Sensitivities.dat')
        fs = sdata[:, 0] * 1e9
        sens = sdata[:, 1]
        template = np.interp(np.log10(self.center_frequencies), np.log10(fs), np.log10(sens), left=-23.126679398184603,
                             right=-21.)
        skysr = 4. * np.pi * (180. / np.pi) ** 2 * self.fsky
        return 10. ** template / np.sqrt(skysr) * np.sqrt(15. / self.duration) * self.mult

    def kpixie_sensitivity(self):
        return r2k(self.center_frequencies, self.pixie_sensitivity())

    def get_function_args(self):
        targs = []
        tp0 = []
        for fnc in self.signals:
            argsp = inspect.getargspec(fnc)
            args = argsp[0][1:]
            p0 = argsp[-1]
            targs = np.concatenate([targs, args])
            tp0 = np.concatenate([tp0, p0])
        return targs, tp0

    def calculate_fisher_matrix(self):
        N = len(self.p0)
        F = np.zeros([N, N])
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
        return (self.measure_signal(**{x: x0 * zp}) - self.measure_signal(**{x: x0})) / (h * x0)

    def measure_signal(self, **kwarg):
        if self.bandpass:
            frequencies = self.band_frequencies
        else:
            frequencies = self.center_frequencies
        N = len(frequencies)
        model = np.zeros(N)
        for fnc in self.signals:
            argsp = inspect.getargspec(fnc)
            args = argsp[0][1:]
            if len(kwarg) and kwarg.keys()[0] in args:
                model += fnc(frequencies, **kwarg)
            else:
                model += fnc(frequencies)
        if self.bandpass:
            return model.reshape((N / self.binstep, self.binstep)).mean(axis=1)
        else:
            return model