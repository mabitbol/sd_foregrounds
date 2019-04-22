import numpy as np
import fisher 
import foregrounds as fg
import spectral_distortions as sd

print "PRISTINE"
fish = fisher.FisherEstimation(duration=12., fmin=82.5e9, priors={'As':0.1, 'alps':0.1})
fish.set_signals(fncs=[sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT,
                       fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad,
                       fg.jens_synch_rad, fg.co_rad])
fish.run_fisher_calculation()
fish.print_errors()

print "PRISTINE"
bx = 120. * (12. / 8760) # months
fish = fisher.FisherEstimation(duration=bx, fmin=82.5e9, priors={'As':0.1, 'alps':0.1})
fish.set_signals(fncs=[sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT,
                       fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad,
                       fg.jens_synch_rad, fg.co_rad])
fish.run_fisher_calculation()
fish.print_errors()

fmin = 82.5e9
p0 = {'As':0.1, 'alps':0.1}
p0 = {}
fsky = 1.
sigs = [sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT, \
        fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad, \
        fg.jens_synch_rad1, fg.co_rad]

print "30 hrs"
bx = 30. * (12. / 8760) # months
fish = fisher.FisherEstimation(duration=bx, fmin=fmin, fsky=fsky, priors=p0)
fish.set_signals(sigs)
fish.run_fisher_calculation()
fish.print_errors()

print "120 hrs"
bx = 120. * (12. / 8760) # months
fish = fisher.FisherEstimation(duration=bx, fmin=fmin, fsky=fsky, priors=p0)
fish.set_signals(sigs)
fish.run_fisher_calculation()
fish.print_errors()


#fish.set_signals(fncs=[sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT, sd.DeltaI_mu,
#                       fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad,
#                       fg.jens_synch_rad, fg.spinning_dust, fg.co_rad])
