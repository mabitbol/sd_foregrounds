import numpy as np
import fisher
import foregrounds as fg
import spectral_distortions as sd


def sens_vs_dnu(fmax=61, fstep=0.1, sens=[1., 0.1, 0.01, 0.001]):
    dnu = np.arange(1, fmax, fstep) * 1.e9
    fish = fisher.FisherEstimation()
    args = fish.args
    N = len(args)
    data = {}
    for sen in sens:
        print "on sens ", sen
        data[sen] = {}
        for arg in args:
            data[sen][arg] = []
        for nu in dnu:
            scale = sen * (15.e9 / nu)
            fish = fisher.FisherEstimation(fstep=nu, mult=scale)
            fish.run_fisher_calculation()
            for arg in args:
                data[sen][arg].append(fish.errors[arg])
    x = {}
    x['sens'] = sens
    x['dnu'] = dnu
    x['data'] = data
    np.save('datatest', x)
    return


def drop_vs_dnu(fmax=61, fstep=0.1, drops=[0, 1, 2]):
    dnu = np.arange(1, fmax, fstep) * 1.e9
    fish = fisher.FisherEstimation()
    args = fish.args
    N = len(args)
    data = {}
    for drop in drops:
        print "on drop ", drop
        data[drop] = {}
        for arg in args:
            data[drop][arg] = []
        for k, nu in enumerate(dnu):
            if k % 100 == 0:
                print "on k ", k 
            scale = 15.e9 / nu
            ps = {'As':0.01, 'alps': 0.01}
            #ps = {'As':0.1, 'alps': 0.1}
            fish = fisher.FisherEstimation(fstep=nu, mult=scale, drop=drop, priors=ps)
            fish.run_fisher_calculation()
            for arg in args:
                data[drop][arg].append(fish.errors[arg])
    x = {}
    x['drops'] = drops
    x['dnu'] = dnu
    x['data'] = data
    np.save('fullcalc_1p_drop012_coarse', x)
    return

def drop_vs_dnu_nomu(fmax=61, fstep=0.1, drops=[0, 1, 2]):
    dnu = np.arange(1, fmax, fstep) * 1.e9
    fish = fisher.FisherEstimation()
    fish.set_signals(fncs=[sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT, 
                           fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad, 
                           fg.jens_synch_rad, fg.spinning_dust, fg.co_rad])
    args = fish.args
    N = len(args)
    data = {}
    for drop in drops:
        print "on drop ", drop
        data[drop] = {}
        for arg in args:
            data[drop][arg] = []
        for k, nu in enumerate(dnu):
            if k % 100 == 0:
                print "on k ", k 
            scale = 15.e9 / nu
            fish = fisher.FisherEstimation(fstep=nu, mult=scale, drop=drop)
            fish.set_signals(fncs=[sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT, 
                                   fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad, 
                                   fg.jens_synch_rad, fg.spinning_dust, fg.co_rad])
            fish.run_fisher_calculation()
            for arg in args:
                data[drop][arg].append(fish.errors[arg])
    x = {}
    x['drops'] = drops
    x['dnu'] = dnu
    x['data'] = data
    np.save('fullcalc_10p_drop012_nomu', x)
    return

def drop_vs_nbin(drops=[0, 1, 2]):
    nbins = np.arange(50, 601)[::-1]
    dnu = 3.e12 / nbins
    fish = fisher.FisherEstimation()
    args = fish.args
    N = len(args)
    data = {}
    for drop in drops:
        print "on drop ", drop
        data[drop] = {}
        for arg in args:
            data[drop][arg] = []
        for k, nu in enumerate(dnu):
            if k % 100 == 0:
                print "on k ", k 
            scale = 15.e9 / nu
            #ps = {'As':0.01, 'alps': 0.01}
            ps = {'As':0.1, 'alps': 0.1}
            fish = fisher.FisherEstimation(fstep=nu, mult=scale, drop=drop, priors=ps)
            fish.run_fisher_calculation()
            for arg in args:
                data[drop][arg].append(fish.errors[arg])
    x = {}
    x['drops'] = drops
    x['dnu'] = dnu
    x['data'] = data
    np.save('fullcalc_10p_drop012_nbins', x)
    return

def drop_vs_nbin_nomu(drops=[0, 1, 2]):
    nbins = np.arange(50, 601)[::-1]
    dnu = 3.e12 / nbins
    fish = fisher.FisherEstimation()
    fish.set_signals(fncs=[sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT, 
                           fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad, 
                           fg.jens_synch_rad, fg.spinning_dust, fg.co_rad])
    args = fish.args
    N = len(args)
    data = {}
    for drop in drops:
        print "on drop ", drop
        data[drop] = {}
        for arg in args:
            data[drop][arg] = []
        for k, nu in enumerate(dnu):
            if k % 100 == 0:
                print "on k ", k 
            scale = 15.e9 / nu
            fish = fisher.FisherEstimation(fstep=nu, mult=scale, drop=drop)
            fish.set_signals(fncs=[sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT, 
                                   fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad, 
                                   fg.jens_synch_rad, fg.spinning_dust, fg.co_rad])
            fish.run_fisher_calculation()
            for arg in args:
                data[drop][arg].append(fish.errors[arg])
    x = {}
    x['drops'] = drops
    x['dnu'] = dnu
    x['data'] = data
    np.save('fullcalc_10p_drop012_nbins_nomu', x)
    return

drop_vs_nbin_nomu()
#drop_vs_dnu(fmax=61, fstep=0.1, drops=[0, 1, 2])
#drop_vs_dnu_nomu(fmax=61, fstep=0.1, drops=[0, 1, 2])



