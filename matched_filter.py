import numpy as np
from scipy import interpolate
import spectral_distortions as sd
import foregrounds as fg
import fisher
from cov_project import cov_inv
import matplotlib
from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])
matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
                  'weight' : 'normal', 'size' : 16}
import matplotlib.pyplot as plt
"""
try a matched filter approach for detecting \mu
"""

# frequency specifications
nu_min = 7.5e9 #7.5e9 (PIXIE)
nu_max = 3000.e9 #3000.e9 (PIXIE)
nu_step = 15.e9 #15.e9 (PIXIE)
fac = 1. #multiplicative scaling of PIXIE noise
fac *= (15.e9/nu_step) #account for multiplicative noise scaling with bin width
#fish = fisher.FisherEstimation(fmin=nu_min, fmax=nu_max, fstep=nu_step, duration=dur, mult=fac, bandpass=False, fsky=1.)

mu_fid = 2.e-8
mu_norm = 1. #for getting mu SED shape only

# use default PIXIE for now
fish = fisher.FisherEstimation(fmin=nu_min, fmax=nu_max, fstep=nu_step, mult=fac, bandpass=True, fsky=0.7)
freqs = fish.center_frequencies
fish.set_signals(fncs=[sd.DeltaI_mu])
# signal template shape
I_mu = fish.measure_signal(mu_amp=mu_norm)

# instrumental noise
noise_PIXIE = fish.pixie_sensitivity()
noise_superPIXIE = np.copy(noise_PIXIE)
noise_superPIXIE[0:7] *= 0.1

# noise covariance
# instrumental noise is diagonal
N_noise = np.diag(noise_PIXIE**2.)
#N_noise = np.diag(noise_superPIXIE**2.)
# CMB blackbody
fish.set_signals(fncs=[sd.DeltaI_DeltaT])
I_CMB = fish.measure_signal(DeltaT_amp=1.2e-5) #true CV = 1.2e-5; our old assumed made-up value was 1.2e-4
N_CMB = np.outer(I_CMB,I_CMB)
I_CMB_pert = fish.measure_signal(DeltaT_amp=1.2e-4) #for testing responses
# mu -- should this be included? not obvious to me
fish.set_signals(fncs=[sd.DeltaI_mu])
I_mu_true = fish.measure_signal(mu_amp=mu_fid)
N_mu = np.outer(I_mu_true,I_mu_true)
# Compton-y
fish.set_signals(fncs=[sd.DeltaI_y])
I_y = fish.measure_signal(y_tot=1.77e-6)
N_y = np.outer(I_y,I_y)
# rtSZ -- includes both Compton-y + relativistic corrections, so don't add N_y and N_rtSZ together (only one at a time)
fish.set_signals(fncs=[sd.DeltaI_reltSZ_2param_yweight])
I_rtSZ = fish.measure_signal(y_tot=1.77e-6, kT_yweight=1.245)
N_rtSZ = np.outer(I_rtSZ,I_rtSZ) #I've checked that multiplying or dividing this by values up to 1.e7 has no effect on mu SNR (in a CMB+rtSZ-only calculation)
I_rtSZ_pert = fish.measure_signal(y_tot=1.77e-6, kT_yweight=1.245 *2.) #for testing responses
# Galactic dust
fish.set_signals(fncs=[fg.thermal_dust_rad])
I_dust = fish.measure_signal(Ad=1.36e6, Bd=1.53, Td=21.)
N_dust = np.outer(I_dust,I_dust)
I_dust_pert = fish.measure_signal(Ad=1.36e6, Bd=1.54, Td=21.) #for testing responses
# CIB
fish.set_signals(fncs=[fg.cib_rad])
I_CIB = fish.measure_signal(Acib=3.46e5, Bcib=0.86, Tcib=18.8)
N_CIB = np.outer(I_CIB,I_CIB)
I_CIB_pert = fish.measure_signal(Acib=3.46e5, Bcib=0.87, Tcib=18.8) #for testing purposes
# CO
fish.set_signals(fncs=[fg.co_rad])
I_CO = fish.measure_signal(Aco=1.)
N_CO = np.outer(I_CO,I_CO)
# free-free
fish.set_signals(fncs=[fg.jens_freefree_rad])
I_ff = fish.measure_signal(EM=300.)
N_ff = np.outer(I_ff,I_ff)
# synchrotron
fish.set_signals(fncs=[fg.jens_synch_rad])
I_synch = fish.measure_signal(As=288., alps=-0.82, w2s=0.2)
N_synch = np.outer(I_synch,I_synch)
I_synch_pert = fish.measure_signal(As=288., alps=-0.83, w2s=0.2) #for testing purposes
# AME
fish.set_signals(fncs=[fg.spinning_dust])
I_AME = fish.measure_signal(Asd=1.)
N_AME = np.outer(I_AME,I_AME)

# sum all contributions
labels = ['noise',
          'noise+CMB',
          'noise+CMB+y/rtSZ',
          'noise+CMB+y/rtSZ+synch',
          'noise+CMB+y/rtSZ+synch+ff',
          'noise+CMB+y/rtSZ+synch+ff+AME',
          'noise+CMB+y/rtSZ+synch+ff+AME+CO',
          'noise+CMB+y/rtSZ+synch+ff+AME+CO+CIB',
          'noise+CMB+y/rtSZ+synch+ff+AME+CO+CIB+dust']
N_vec = np.array([N_noise,
                  N_noise + N_CMB,
                  N_noise + N_CMB + N_rtSZ,
                  N_noise + N_CMB + N_rtSZ + N_synch,
                  N_noise + N_CMB + N_rtSZ + N_synch + N_ff,
                  N_noise + N_CMB + N_rtSZ + N_synch + N_ff + N_AME,
                  N_noise + N_CMB + N_rtSZ + N_synch + N_ff + N_AME + N_CO,
                  N_noise + N_CMB + N_rtSZ + N_synch + N_ff + N_AME + N_CO + N_CIB,
                  N_noise + N_CMB + N_rtSZ + N_synch + N_ff + N_AME + N_CO + N_CIB + N_dust])
I_arr_vec = np.array([[I_mu],
                      [I_mu,I_CMB],
                      [I_mu,I_CMB,I_rtSZ],
                      [I_mu,I_CMB,I_rtSZ,I_synch],
                      [I_mu,I_CMB,I_rtSZ,I_synch,I_ff],
                      [I_mu,I_CMB,I_rtSZ,I_synch,I_ff,I_AME],
                      [I_mu,I_CMB,I_rtSZ,I_synch,I_ff,I_AME,I_CO],
                      [I_mu,I_CMB,I_rtSZ,I_synch,I_ff,I_AME,I_CO,I_CIB],
                      [I_mu,I_CMB,I_rtSZ,I_synch,I_ff,I_AME,I_CO,I_CIB,I_dust]])

#N = N_noise + N_CIB + N_dust
#N = N_noise + N_synch + N_ff + N_AME + N_CO + N_CMB + N_rtSZ
#N = N_noise + N_CMB + N_rtSZ + N_dust + N_CIB + N_CO + N_ff + N_synch + N_AME

# matched filter noise rms on signal amplitude
def sigma_MF(temp,N_cov): #temp=template SED, N_cov=cov matrix
    assert N_cov.shape[0] == N_cov.shape[1]
    assert N_cov.shape[0] == len(temp)
    N_cov_inv = np.linalg.inv(N_cov) #caution...
    #print np.amax( np.absolute( np.inner(N_cov,N_cov_inv) - np.identity(len(temp)) ) )
    return 1. / np.sqrt(np.dot( np.dot(temp,N_cov_inv), temp ))
# matched filter noise rms on signal amplitude -- handle difficult cov matrix inversion with mode deprojection technique
def sigma_MF_alt(temp,N_cov): #temp=template SED, N_cov=cov matrix
    assert N_cov.shape[0] == N_cov.shape[1]
    assert N_cov.shape[0] == len(temp)
    #N_cov_inv = np.linalg.inv(N_cov) #caution...
    [N_cov_inv, N_deproj] = cov_inv(N_cov,len(temp))
    #print np.amax( np.absolute( np.inner(N_cov,N_cov_inv) - np.identity(len(temp)) ) ), N_deproj
    return 1. / np.sqrt(np.dot( np.dot(temp,N_cov_inv), temp ))

#mu_SNR_noiseonly = mu_fid / sigma_MF(I_mu, N_noise)
#print mu_SNR_noiseonly
#mu_SNR = mu_fid / sigma_MF(I_mu, N)
#print mu_SNR
#quit()

# matched filter weights
def w_MF(temp,N_cov):
    assert N_cov.shape[0] == N_cov.shape[1]
    assert N_cov.shape[0] == len(temp)
    N_cov_inv = np.linalg.inv(N_cov) #caution...
    return np.dot(temp,N_cov_inv) / np.dot( np.dot(temp,N_cov_inv), temp )
# matched filter weights -- handle difficult cov matrix inversion with mode deprojection technique
def w_MF_alt(temp,N_cov):
    assert N_cov.shape[0] == N_cov.shape[1]
    assert N_cov.shape[0] == len(temp)
    #N_cov_inv = np.linalg.inv(N_cov) #caution...
    [N_cov_inv, N_deproj] = cov_inv(N_cov,len(temp))
    return np.dot(temp,N_cov_inv) / np.dot( np.dot(temp,N_cov_inv), temp )

# noise check, given weights
def sigma_MF_check(w,N_cov):
    return np.sqrt( np.dot( np.dot(w,N_cov), w) )

# response of weights to some component
def MF_resp(w,temp):
    return np.dot(w,temp)

# weighted least-squares linear regression weights, for comparison
def w_LR(I_arr, N_n):
    N_noise_inv = np.linalg.inv(N_n)
    N_comp = len(I_arr)
    # build C_ab matrix
    C_ab = np.zeros((N_comp,N_comp))
    for a in range(N_comp):
        for b in range(a,N_comp):
            C_ab[a][b] = np.dot( np.dot(I_arr[a],N_noise_inv), I_arr[b])
            if b!=a:
                C_ab[b][a] = C_ab[a][b]
    # invert
    C_ab_inv = np.linalg.inv(C_ab)
    # compute weights -- this is a matrix with weights for each component in I_arr
    return np.dot( C_ab_inv, np.dot(I_arr, N_noise_inv) )


# results
num_N = len(N_vec)
sigma_MF_vec = np.zeros(num_N)
mu_SNR_vec = np.zeros(num_N)
sigma_MF_alt_vec = np.zeros(num_N)
mu_SNR_alt_vec = np.zeros(num_N)
sigma_MF_check_vec = np.zeros(num_N)
mu_SNR_check_vec = np.zeros(num_N)
plt.clf()
plt.xlim(nu_min/1.e9,nu_max/1.e9)
#plt.ylim()
plt.xlabel(r'$\nu \, [{\rm GHz}]$',fontsize=17)
plt.ylabel(r'$w(\nu) \, [({\rm Jy/sr})^{-1}]$',fontsize=17)
plt.grid(alpha=0.5)
for i in range(num_N):
    print "---------------------------------"
    print i,labels[i]
    sigma_MF_vec[i] = sigma_MF(I_mu, N_vec[i])
    sigma_MF_alt_vec[i] = sigma_MF_alt(I_mu, N_vec[i])
    mu_SNR_vec[i] = mu_fid / sigma_MF_vec[i]
    mu_SNR_alt_vec[i] = mu_fid / sigma_MF_alt_vec[i]
    weights = w_MF(I_mu, N_vec[i])
    sigma_MF_check_vec[i] = sigma_MF_check(weights,N_vec[i])
    mu_SNR_check_vec[i] = mu_fid / sigma_MF_check_vec[i]
    print "SNR:",mu_SNR_vec[i],mu_SNR_alt_vec[i],mu_SNR_check_vec[i]
    print "resp (mu):",MF_resp(weights,I_mu_true)
    print "resp (CMB):",MF_resp(weights,I_CMB)
    #print "resp (CMB perturbed):",MF_resp(weights,I_CMB_pert)
    #print "resp (y):",MF_resp(weights,I_y)
    print "resp (rtSZ):",MF_resp(weights,I_rtSZ)
    print "resp (rtSZ perturbed):",MF_resp(weights,I_rtSZ_pert)
    print "resp (dust):",MF_resp(weights,I_dust)
    print "resp (dust perturbed):",MF_resp(weights,I_dust_pert)
    print "resp (CIB):",MF_resp(weights,I_CIB)
    print "resp (CIB perturbed):",MF_resp(weights,I_CIB_pert)
    print "resp (CO):",MF_resp(weights,I_CO)
    print "resp (ff):",MF_resp(weights,I_ff)
    print "resp (AME):",MF_resp(weights,I_AME)
    print "resp (synch):",MF_resp(weights,I_synch)
    print "resp (synch perturbed):",MF_resp(weights,I_synch_pert)
    plt.semilogx(freqs/1.e9, weights, label=labels[i])
    #plt.semilogx(freqs/1.e9, w_MF_alt(I_mu, N_vec[i]), ls='--', alpha=0.7)
    weights_LR = (w_LR(I_arr_vec[i], N_noise))[0]
    plt.semilogx(freqs/1.e9, weights_LR, alpha=0.7, ls='--', label=labels[i]+' (LR)')
    print np.amax((weights-weights_LR)/weights)
    print np.argmax((weights-weights_LR)/weights)
    print freqs[np.argmax((weights-weights_LR)/weights)]
plt.axhline(y=0.,color='k',lw=0.75)
plt.legend(loc='lower right',fontsize=4,ncol=1)
#plt.savefig('matched_filter_weights.pdf')
plt.savefig('matched_filter_vs_LR_weights.pdf')
