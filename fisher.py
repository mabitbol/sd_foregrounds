import numpy as np
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


def get_normcov(fncs, freqs, errs):
    args, p0 = fncs_args(fncs)
    F = fisher_signals(signals, freqs, fncs, args, p0, errs)
    cov = np.mat(F).I
    N = cov.shape[0]
    for k in range(N):
        print "fisher uncertainty on %s is %f percent" %(args[k], np.sqrt(cov[k,k]) / p0[k] * 100)
    normcov = np.zeros([N,N])
    for i in range(N):
        for k in range(N):
            normcov[i,k] = cov[i,k] / (p0[i] * p0[k])
    return normcov, args, p0
