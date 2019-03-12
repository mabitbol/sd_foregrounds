import fisher 
import foregrounds as fg
import spectral_distortions as sd

# PIXIE with all signals is the default. 
fish = fisher.FisherEstimation()
# args are stored in fish.args, values are stored in dictionary fish.argvals, 
# and fisher uncertainties are in fish.errors
fish.run_fisher_calculation()
# print the errors in sigma
fish.print_errors()

# To set the signals by hand, just modify the fncs arg here:
fish = fisher.FisherEstimation()
fish.set_signals(fncs=[sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT, sd.DeltaI_mu,
                       fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad,
                       fg.jens_synch_rad, fg.spinning_dust, fg.co_rad])
fish.run_fisher_calculation()
fish.print_errors()


# To change the frequencies (Hz), duration (months), or scale the noise by mult, 
# turn off the step function bandpass or change fsky, 
# edit any of the following
fish = fisher.FisherEstimation(fmin=5.e9, fmax=1.e12, fstep=5.e9, duration=60, mult=0.1, bandpass=False, fsky=0.5)


# Lastly to put priors (in fractions of the parameter value), drop the first n bins or mask out Galactic CO lines do:
fish = fisher.FisherEstimation(priors={'Td':0.1, 'Asd':0.01}, drop=2, doCO=True) 

