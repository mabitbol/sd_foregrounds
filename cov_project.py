import numpy as np

"""
deprojection of modes corresponding to extremely small eigenvalues when inverting a cov matrix
originally suggested in email from Kendrick Smith on 2 March 2013 -- minor corrections to the equations there (want R = V D V^T)
relies on R being a real, symmetric matrix, as it is if C is a covariance matrix
"""

### threshold parameter for deprojecting ###
thresh = 1.0e15
###

def cov_inv(cov,dim): #cov = dim x dim -sized cov matrix
    Delta = np.diag(cov)
    # construct correlation matrix R_ij = C_ij/sqrt(C_ii C_jj)
    R = np.zeros((dim,dim))
    for i in xrange(dim):
        for j in xrange(dim):
            R[i][j] = cov[i][j] / np.sqrt(cov[i][i]*cov[j][j])
            R[j][i] = R[i][j] #symmetrize
    # compute eigenvalues and eigenvectors of R
    eigenValues,eigenVectors = np.linalg.eig(R)
    # sort from largest to smallest eigenvalue
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    # deproject any eigenvalues that are a factor of thresh smaller than the largest one
    eigenValues_inv = 1.0/eigenValues
    N_deproj = 0 #count how many are deprojected
    for i in xrange(1,dim):
        if (eigenValues[0]/eigenValues[i] > thresh):
            eigenValues_inv[i] = 0.0
            N_deproj += 1
    # compute cov^-1 after having deprojected the modes associated with these eigenvalues
    R_inv = np.inner(np.inner(eigenVectors,np.diag(eigenValues_inv)),eigenVectors)
    Delta_fac = np.diag( 1.0/np.sqrt(Delta) )
    cov_inv = np.inner(np.inner(Delta_fac,R_inv),Delta_fac)
    return [cov_inv, N_deproj]
