import numpy as np
import sys
import inspect
import sympy as sym
import signals 

class Fisher:
    def __init__(self, x, signals, noise):
        self.x = x
        self.lenx = len(x)
        self.signals = signals
        self.noise = noise
        
        self.setup_signals()
    
    def setup_signals(self):
        # do something get args, values, etc
        
    def calculate_sed_derivative_matrix(self):
        smat = np.zeros(( self.nargs, self.nargs, self.lenx ))
        for i in range(self.nargs):
            dfdpi = self.signal_derivative(self.args[i], self.p0[i])
            for j in range(self.nargs):
                dfdpj = self.signal_derivative(self.args[j], self.p0[j])
                smat[i, j] = dfdpi * dfdpj
        self.sed_derivative_matrix = smat

    def calculate_fisher_matrix(self):
        inverse_variance = 1. / (self.noise * self.noise) # check if this is shitty? 
        Fmat = np.zeros(( self.nargs, self.nargs ))
        for i in range(self.nargs):
            for j in range(self.nargs):
                Fmat[i, j] = np.dot( self.sed_derivative_matrix[i, j], inverse_variance )
        self.Fmat = np.mat(Fmat)

    def calculate_covariance(self, check_good=True):
        if check_good: 
            if np.linalg.cond(self.Fmat) > (1. / sys.float_info.epsilon):
                print "bad"
                u, s, vh = np.linalg.svd(self.Fmat, full_matrices=True)
                # etc
                cov = 1
            else:
                cov = np.linalg.solve(self.Fmat, np.identity(self.nargs))
        else: 
            cov = np.linalg.solve(self.Fmat, np.identity(self.nargs))
        return cov



