import numpy as np
import warnings

def distribution_normal(R,Rm,sigma): #R to evaluate, Mean radius and standard deviation of distribution
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(R-Rm)**2/(2*sigma**2))

def distribution_lognormal(R,Rm,sigma): # converted lognormal distribution with true mean radius and standard deviation
    Rm_parameter = np.log((Rm**2)/np.sqrt(Rm**2+sigma**2))
    sigma_parameter = np.sqrt(np.log(1+(sigma**2)/Rm**2))
    result = 1/(np.sqrt(2*np.pi)*sigma_parameter*R)*np.exp(-(np.log(R)-Rm_parameter)**2/(2*(sigma_parameter**2)))
    return result
