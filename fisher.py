import numpy as np
import foregrounds as fg
import components2 as sd
import inspect

def deriv_signals(f, freqs, fncs, x, x0):
    h = 1.e-4
    zp = 1. + h
    return ( f(freqs, fncs, **{x:x0*zp}) - f(freqs, fncs, **{x:x0}) ) / (h*x0)

def fncs_args(fncs):
    targs = []
    tp0 = []
    for fnc in fncs:
        argsp = inspect.getargspec(fnc)
        args = argsp[0][1:]
        p0 = argsp[-1]
        targs = np.concatenate([targs, args])
        tp0 = np.concatenate([tp0, p0])
    return targs, tp0

def signals(freqs, fncs, **kwarg):
    model = np.zeros(len(freqs))
    for fnc in fncs:
        argsp = inspect.getargspec(fnc)
        args = argsp[0][1:]
        if len(kwarg) and kwarg.keys()[0] in args:
            model += fnc(freqs, **kwarg)
        else:
            model += fnc(freqs)
    return model

def fisher_signals(signals, freqs, fncs, args, p0, sigmas):
    N = len(p0)
    F = np.zeros([N, N])
    for i in range(N):
        dfdpi = deriv_signals(signals, freqs, fncs, args[i], p0[i])
        dfdpi /= sigmas
        for j in range(N):
            dfdpj = deriv_signals(signals, freqs, fncs, args[j], p0[j])
            dfdpj /= sigmas

            F[i,j] = np.dot(dfdpi, dfdpj)
    return F

def get_normcov(fncs, freqs, errs, write=True):
    args, p0 = fncs_args(fncs)
    F = fisher_signals(signals, freqs, fncs, args, p0, errs)
    cov = np.mat(F).I
    N = cov.shape[0]
    if write:
        for k in range(N):
            print "fisher uncertainty on %s is %f percent" %(args[k], np.sqrt(cov[k,k]) / p0[k] * 100)
    normcov = np.zeros([N,N])
    for i in range(N):
        for k in range(N):
            normcov[i,k] = cov[i,k] / (p0[i] * p0[k])
    return cov, normcov, args, p0

def get_fisher(fncs, freqs, errs):
    args, p0 = fncs_args(fncs)
    F = fisher_signals(signals, freqs, fncs, args, p0, errs)
    cov = np.mat(F).I
    return F, cov, args, p0

def get_cmb_sigma(fmin=15.e9, fmax=3.e12, fstep=15.e9, length=15., prior=0.01, mult=1.):
    freqs = fg.pixie_frequencies(fmin=fmin, fmax=fmax, fstep=fstep)
    noisek = sd.kpixie_sensitivity(freqs, fsky=0.7) * np.sqrt(15./length) * mult
    fncs = [sd.kDeltaI_mu, sd.kDeltaI_reltSZ_2param_yweight, sd.kDeltaI_DeltaT, fg.jens_freefree, \
                fg.jens_synch, fg.cib, fg.ame, fg.co]
    F, cov, args, p0 = get_fisher(fncs, freqs, noisek)
    F[5,5] = 1e60  
    if prior > 0:
        F[7,7] += 1./(prior * p0[7])**2
    cov = np.mat(F).I 
    return F, cov, args, p0

def get_2x2cov(cov, args, p0, names):
    index = []
    ncov = []
    for name in names:
        index.append(np.where(args==name)[0][0])
    for i in index:
        for k in index:
            ncov.append(cov[i,k])
    return np.array(ncov).reshape(2,2), args[index], p0[index]


