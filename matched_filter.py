import numpy as np
from scipy import interpolate
import spectral_distortions as sd
import foregrounds as fg
import fisher
"""
try a matched filter approach for detecting \mu
"""

mu_fid = 2.e-8
mu_norm = 1. #for getting mu SED shape only

# use default PIXIE for now
fish = fisher.FisherEstimation()
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
I_CMB = fish.measure_signal(DeltaT_amp=1.2e-4)
N_CMB = np.outer(I_CMB,I_CMB)
# Compton-y
fish.set_signals(fncs=[sd.DeltaI_y])
I_y = fish.measure_signal(y_tot=1.77e-6)
N_y = np.outer(I_y,I_y)
# rtSZ -- includes both Compton-y + relativistic corrections, so don't add N_y and N_rtSZ together (only one at a time)
fish.set_signals(fncs=[sd.DeltaI_reltSZ_2param_yweight])
I_rtSZ = fish.measure_signal(y_tot=1.77e-6, kT_yweight=1.245)
N_rtSZ = np.outer(I_rtSZ,I_rtSZ)
# Galactic dust
fish.set_signals(fncs=[fg.thermal_dust_rad])
I_dust = fish.measure_signal(Ad=1.36e6, Bd=1.53, Td=21.)
N_dust = np.outer(I_dust,I_dust)
# CIB
fish.set_signals(fncs=[fg.cib_rad])
I_CIB = fish.measure_signal(Acib=3.46e5, Bcib=0.86, Tcib=18.8)
N_CIB = np.outer(I_CIB,I_CIB)
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
# AME
fish.set_signals(fncs=[fg.spinning_dust])
I_AME = fish.measure_signal(Asd=1.)
N_AME = np.outer(I_AME,I_AME)

# sum all contributions
#N = N_noise + N_CMB + N_y
N = N_noise + N_CMB + N_rtSZ + N_dust + N_CIB + N_CO + N_ff + N_synch + N_AME



# matched filter noise rms on signal amplitude
def sigma_MF(temp,N_cov):
    assert N_cov.shape[0] == N_cov.shape[1]
    assert N_cov.shape[0] == len(temp)
    N_cov_inv = np.linalg.inv(N_cov) #caution...
    return 1. / np.sqrt(np.dot( np.dot(temp,N_cov_inv), temp ))

mu_SNR = mu_fid / sigma_MF(I_mu, N)
print mu_SNR
mu_SNR_noiseonly = mu_fid / sigma_MF(I_mu, N_noise)
print mu_SNR_noiseonly
