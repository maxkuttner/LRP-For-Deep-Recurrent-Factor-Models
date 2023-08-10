import numpy as np
from numpy import newaxis as na

def lrp_linear(w, b, z_i, z_j, Rj, nlower, eps=1e-4, delta=0.0):
    """
    LRP for a linear layer with input (previous layer) dim D and output (next layer) dim M.
    Args:
    
    w:   weights from layer i (lower) to j (higher) - array of shape (D, M)
    b:   biases  from layer i (lower) to j (higher) - array of shape (M, )
    z_i: linear activation of node i in lower layer - array of shape (D, )
    z_j: linear activation of node j in upper layer - array of shape (M, )
    Rj:  relevance score of node j from upper layer - array of shape (M, )
    nlower: the number of nodes in the lower layer  - this will be 
    eps: correction error for stabilisation i.e. to avoid cases like 0/0
    delta: set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
    
    Returns:
    
    Ri: relevance score for lower layer node - array of shape (D, )
    
    '''
    @author: Leila Arras
    @maintainer: Leila Arras
    @date: 21.06.2017
    @version: 1.0+
    @copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
    @license: see LICENSE file in repository root
    '''
    """
    
    sign_out = np.where(z_j[na,:]>=0, 1., -1.) # shape (1, M)
    
    # define the numerator
    numer = (w * z_i[:,na]) + ( delta * (b[na,:] + eps * sign_out ) / nlower ) # shape (D, M)
    
    # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
    # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
    # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)
    
    denom = z_j[na,:] + (eps*sign_out*1.)   # shape (1, M)
    
    message = (numer/denom) * Rj[na,:]       # shape (D, M)
    
    Rin = message.sum(axis=1)              # shape (D,)

    return Rin

