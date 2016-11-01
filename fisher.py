import numpy as np
import foregrounds as fg
import components as sd
import inspect

def deriv_signals(f, freqs, fncs, x, x0):
    h = 1.e-4
    zp = 1. + h
    return ( f(freqs, fncs, **{x:x0*zp}) - f(freqs, fncs, **{x:x0}) ) / (h*x0)

def deriv_signals_bandpass(f, freqs, fncs, x, x0, binstep):
    h = 1.e-4
    zp = 1. + h
    return ( f(freqs, fncs, binstep, **{x:x0*zp}) - f(freqs, fncs, binstep, **{x:x0}) ) / (h*x0)

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

def signals_bandpass(freqs, fncs, binstep, **kwarg):
    N = len(freqs)/binstep
    model = np.zeros(N)
    for fnc in fncs:
        argsp = inspect.getargspec(fnc)
        args = argsp[0][1:]
        if len(kwarg) and kwarg.keys()[0] in args:
            hold = fnc(freqs, **kwarg).reshape((N, binstep)).mean(axis=1)
            model += fnc(freqs, **kwarg).reshape((N, binstep)).mean(axis=1)
        else:
            model += fnc(freqs).reshape((N, binstep)).mean(axis=1)
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

def fisher_signals_bandpass(signals_bandpass, freqs, fncs, args, p0, sigmas, binstep):
    N = len(p0)
    F = np.zeros([N, N])
    for i in range(N):
        dfdpi = deriv_signals_bandpass(signals_bandpass, freqs, fncs, args[i], p0[i], binstep)
        dfdpi /= sigmas
        for j in range(N):
            dfdpj = deriv_signals_bandpass(signals_bandpass, freqs, fncs, args[j], p0[j], binstep)
            dfdpj /= sigmas
            F[i,j] = np.dot(dfdpi, dfdpj)
    return F

def get_fisher(fncs, freqs, errs):
    args, p0 = fncs_args(fncs)
    F = fisher_signals(signals, freqs, fncs, args, p0, errs)
    cov = np.mat(F).I
    return F, cov, args, p0

def get_fisher_bandpass(fncs, freqs, errs, binstep):
    args, p0 = fncs_args(fncs)
    F = fisher_signals_bandpass(signals_bandpass, freqs, fncs, args, p0, errs, binstep)
    cov = np.mat(F).I
    return F, cov, args, p0

def get_covariance_wbandpass(fmin=15.e9, fmax=3.e12, fstep=15.e9, length=15., prior=0.01, mult=1., \
                             fncs=[sd.kDeltaI_mu, sd.kDeltaI_reltSZ_2param_yweight, sd.kDeltaI_DeltaT, \
                                   fg.jens_freefree1p, fg.jens_synch, fg.cib, fg.spinning_dust, fg.co]
                             ):
    freqs, centerfreqs, binstep = band_averaging_freqs(fmin, fmax, fstep)
    noise = sd.kpixie_sensitivity(centerfreqs, fsky=0.7) * np.sqrt(15./length) * mult
    F, cov, args, p0 = get_fisher_bandpass(fncs, freqs, noise, binstep)
    if prior > 0:
        alps_index = np.where(args=='alps')[0][0]
        F[alps_index,alps_index] += 1./(prior * p0[alps_index])**2
    N = len(args)
    normF = np.zeros([N,N])
    for i in range(N):
        normF[i,i] = 1./F[i,i]
    cov =  (np.mat(normF)*np.mat(F)).I * np.mat(normF) 
    return F, cov, args, p0, centerfreqs, noise

def get_cmb_sigma(fmin=15.e9, fmax=3.e12, fstep=15.e9, length=15., prior=0.01, mult=1.):
    freqs = sd.pixie_frequencies(fmin=fmin, fmax=fmax, fstep=fstep)
    noisek = sd.kpixie_sensitivity(freqs, fsky=0.7) * np.sqrt(15./length) * mult
    fncs = [sd.kDeltaI_mu, sd.kDeltaI_reltSZ_2param_yweight, sd.kDeltaI_DeltaT, \
                fg.jens_freefree1p, fg.jens_synch, fg.cib, fg.spinning_dust, fg.co]
    F, cov, args, p0 = get_fisher(fncs, freqs, noisek)
    #F[5,5] = 1e60  
    if prior > 0:
        alps_index = np.where(args=='alps')[0][0]
        F[alps_index,alps_index] += 1./(prior * p0[alps_index])**2
    N = len(args)
    normF = np.zeros([N,N])
    for i in range(N):
        normF[i,i] = 1./F[i,i]
    cov =  (np.mat(normF)*np.mat(F)).I * np.mat(normF) 
    return F, cov, args, p0, freqs, noisek

def band_averaging_freqs(fmin, fmax, fstep):
    freqs = np.arange(fmin, fmax+fstep, 1.e9)
    binstep = int(fstep/1.e9)
    freqs = freqs[:(len(freqs)/binstep)*binstep]
    centerfreqs = freqs.reshape((len(freqs)/binstep, binstep)).mean(axis=1)
    return freqs, centerfreqs, binstep

def get_2x2cov(cov, args, p0, names):
    index = []
    ncov = []
    for name in names:
        index.append(np.where(args==name)[0][0])
    for i in index:
        for k in index:
            ncov.append(cov[i,k])
    return np.array(ncov).reshape(2,2), args[index], p0[index]

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

