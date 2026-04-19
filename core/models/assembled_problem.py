from scipy.integrate import quad
import numpy as np
"""
Functions for handling scattering intensity computation: large scatterer approximation + background, and nano contribution only.
When fitting systems with two distributions, use the final_intensity version only once
"""

"""
nano_intensity_spheroid: Function for determining scattering intensity by nanoscale scatterers, applicable to sphere-like scatterers
Overloaded function for all intensity computation, mode is determined by number of inputs. Returns intensity in cm-1
combinedfactor: volume fraction * contrast factor^2(nm-2) * 10^7
distribution: distribution function of the scatterer. If not a function, then final_intensity switches to monodistribution
formfactor: form factor of the scatterer
volume: volume function of the scatterer
background: background signal
Q: scattering vector, nm-1
R: radius/characteristic length of scatterer, nm
sigma: std.dev. of scatterer characteristic length, nm
k: major/minor axis ratio, or core/shell ratio, depending on the config
mu: ratio of core/shell contrast factor
"""
def nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,Q,R,sigma=None,k=None,mu=None):
    if callable(distribution): #distribution with density
        if k is None: #sphere
            integrand_numerator = lambda Re: formfactor(Q,Re)**2*volume(Re)**2*distribution(Re,R,sigma)
            numerator,_ = quad(integrand_numerator,0.001,50,points=[R])
            integrand_denominator = lambda Re: volume(Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50,points=[R])
            I_nano = combinedfactor*numerator/denominator
            return I_nano
        elif mu is None: #ellipsoid
            integrand_numerator = lambda Re: formfactor(Q,Re,k*Re)**2*volume(Re,k*Re)**2*distribution(Re,R,sigma)
            numerator,_ = quad(integrand_numerator,0.001,50,points=[R])
            integrand_denominator = lambda Re: volume(Re,k*Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50,points=[R])
            I_nano = combinedfactor*numerator/denominator
            return I_nano
        else: #core-shell
            integrand_numerator = lambda Re: formfactor(Q,Re,k,mu)**2*volume(Re)**2*distribution(Re,R,sigma)
            numerator,_ = quad(integrand_numerator,0.001,50,points=[R])
            integrand_denominator = lambda Re: volume(Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50,points=[R])
            I_nano = combinedfactor*numerator/denominator
            return I_nano 
    else: #mono distribution
        if k is None: #sphere
            volume_scatterer = volume(R)
            formfactor_scatterer = formfactor(Q,R)
        elif mu is None: #ellipsoid
            volume_scatterer = volume(R,k*R)
            formfactor_scatterer = formfactor(Q,R,k*R)
        else: #core-shell
            volume_scatterer = volume(R)
            formfactor_scatterer = formfactor(Q,R,k,mu)
        return combinedfactor*volume_scatterer*formfactor_scatterer**2

"""
final_intensity_spheroid: Master function for determining scattering intensity at a certain scattering vector Q, applicable to sphere-like scatterers
Overloaded function for all intensity computation, mode is determined by number of inputs. Returns intensity in cm-1
approx: function for residual large scatterer signal approximation, by default porod region
combinedfactor: volume fraction * contrast factor^2(nm-2) * 10^7
distribution: distribution function of the scatterer. If not a function, then final_intensity switches to monodistribution
formfactor: form factor of the scatterer
volume: volume function of the scatterer
background: background signal
Q: scattering vector, nm-1
R: radius/characteristic length of scatterer, nm
C: constant in approximation function
sigma: std.dev. of scatterer characteristic length, nm
k: major/minor axis ratio, or core/shell ratio, depending on the config
mu: ratio of core/shell contrast factor
gamma: fractal dimension of approximation function.
"""

def final_intensity_spheroid(approx,combinedfactor,distribution,formfactor,volume,background,Q,R,C,gamma=4,sigma=None,k=None,mu=None):
    I_porod = approx(Q,gamma,C)
    if callable(distribution): #distribution with density
        if k is None: #sphere
            integrand_numerator = lambda Re: formfactor(Q,Re)**2*volume(Re)**2*distribution(Re,R,sigma)
            numerator,_ = quad(integrand_numerator,0.001,50,points=[R])
            integrand_denominator = lambda Re: volume(Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50,points=[R])
            I_nano = combinedfactor*numerator/denominator
            return I_porod + I_nano + background
        elif mu is None: #ellipsoid
            integrand_numerator = lambda Re: formfactor(Q,Re,k*Re)**2*volume(Re,k*Re)**2*distribution(Re,R,sigma)
            numerator,_ = quad(integrand_numerator,0.001,50,points=[R])
            integrand_denominator = lambda Re: volume(Re,k*Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50,points=[R])
            I_nano = combinedfactor*numerator/denominator
            return I_porod + I_nano + background
        else: #core-shell
            integrand_numerator = lambda Re: formfactor(Q,Re,k,mu)**2*volume(Re)**2*distribution(Re,R,sigma)
            numerator,_ = quad(integrand_numerator,0.001,50,points=[R])
            integrand_denominator = lambda Re: volume(Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50,points=[R])
            I_nano = combinedfactor*numerator/denominator
            return I_porod + I_nano + background
    else: #mono distribution
        if k is None: #sphere
            volume_scatterer = volume(R)
            formfactor_scatterer = formfactor(Q,R)
        elif mu is None: #ellipsoid
            volume_scatterer = volume(R,k*R)
            formfactor_scatterer = formfactor(Q,R,k*R)
        else: #core-shell
            volume_scatterer = volume(R)
            formfactor_scatterer = formfactor(Q,R,k,mu)
        return I_porod + combinedfactor*volume_scatterer*formfactor_scatterer**2 + background
"""
single_intensity: intensity value with scale conversion
"""  
"""
def single_intensity_spheroid(approx,
                              combinedfactor_log,
                              distribution,
                              formfactor,
                              volume,
                              background_log,
                              Q,
                              R,
                              C_log,
                              gamma=4,
                              sigma=None,
                              k=None,
                              mu=None):
    combinedfactor = np.exp(-combinedfactor_log)
    background = np.exp(background_log)
    C = np.exp(C_log)
    sigma_rm = None
    if sigma is not None:
        sigma_rm = sigma*R
    return final_intensity_spheroid(approx,combinedfactor,distribution,formfactor,volume,background,Q,R,C,gamma=gamma,sigma=sigma_rm,
                                   k=k,mu=mu)
"""
def single_intensity_spheroid(approx,
                              combinedfactor_log,
                              distribution,
                              formfactor,
                              volume,
                              Q,
                              R,  
                              gamma=4,
                              background_log=None,
                              C_log=None,
                              sigma=None,
                              k=None,
                              mu=None):
    combinedfactor = np.exp(-combinedfactor_log)
    
    sigma_rm = None
    if sigma is not None:
        sigma_rm = sigma*R
    if background_log is not None:
        background = np.exp(background_log)
        C = np.exp(C_log)
        return final_intensity_spheroid(approx,combinedfactor,distribution,formfactor,volume,background,Q,R,C,gamma=gamma,sigma=sigma_rm,
                                   k=k,mu=mu)
    else:
        return nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,Q,R,sigma=sigma_rm,
                                   k=k,mu=mu)
"""
double_intensity: combined value with scale conversion
"""
def double_intensity_spheroid(approx,
                              combinedfactor_log,
                              combinedfactor_2_log,
                              distribution,
                              distribution_2,
                              formfactor,
                              formfactor_2,
                              volume,
                              volume_2,
                              Q,
                              R,
                              R_2,
                              gamma=4,
                              background_log=None,
                              C_log=None,
                              sigma=None,
                              sigma_2=None,
                              k=None,
                              k_2=None,
                              mu=None,
                              mu_2=None):
    combinedfactor = np.exp(-combinedfactor_log)
    combinedfactor_2 = np.exp(-combinedfactor_2_log)
    sigma_rm = None
    sigma_rm_2 = None
    if sigma is not None:
        sigma_rm = sigma*R
    if sigma_2 is not None:
        sigma_rm_2 = sigma_2*R_2
    if background_log is not None:
        background = np.exp(background_log)
        C = np.exp(C_log)
        res = final_intensity_spheroid(approx,combinedfactor,distribution,formfactor,volume,background,Q,R,C,gamma=gamma,sigma=sigma_rm,
                                    k=k,mu=mu) + nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,Q,R_2,
                                                                                sigma=sigma_rm_2,k=k_2,mu=mu_2)
    else:
        res = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,Q,R,sigma=sigma_rm,
                                    k=k,mu=mu) + nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,Q,R_2,
                                                                                sigma=sigma_rm_2,k=k_2,mu=mu_2)
    return res
"""
mag specific functions such that parameter names are different(very important)
"""
def single_intensity_spheroid_mag(approx,
                              combinedfactor_log_mag,
                              distribution,
                              formfactor,
                              volume,
                              Q,
                              R,
                              gamma=4,
                              background_log_mag=None,
                              C_log_mag=None,
                              sigma=None,
                              k=None,
                              mu_mag=None):
    combinedfactor = np.exp(-combinedfactor_log_mag)
    
    sigma_rm = None
    if sigma is not None:
        sigma_rm = sigma*R
    if background_log_mag is not None:
        background = np.exp(background_log_mag)
        C = np.exp(C_log_mag)
        return final_intensity_spheroid(approx,combinedfactor,distribution,formfactor,volume,background,Q,R,C,gamma=gamma,sigma=sigma_rm,
                                   k=k,mu=mu_mag)
    else:
        return nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,Q,R,sigma=sigma_rm,
                                   k=k,mu=mu_mag)

def double_intensity_spheroid_mag(approx,
                              combinedfactor_log_mag,
                              combinedfactor_2_log_mag,
                              distribution,
                              distribution_2,
                              formfactor,
                              formfactor_2,
                              volume,
                              volume_2,
                              
                              Q,
                              R,
                              R_2,
                              
                              gamma=4,
                              background_log_mag=None,
                              C_log_mag=None,
                              sigma=None,
                              sigma_2=None,
                              k=None,
                              k_2=None,
                              mu_mag=None,
                              mu_2_mag=None):
    combinedfactor = np.exp(-combinedfactor_log_mag)
    combinedfactor_2 = np.exp(-combinedfactor_2_log_mag)
    
    sigma_rm = None
    sigma_rm_2 = None
    if sigma is not None:
        sigma_rm = sigma*R
    if sigma_2 is not None:
        sigma_rm_2 = sigma_2*R_2
    if background_log_mag is not None:
        background = np.exp(background_log_mag)
        C = np.exp(C_log_mag)
        res = final_intensity_spheroid(approx,combinedfactor,distribution,formfactor,volume,background,Q,R,C,gamma=gamma,sigma=sigma_rm,
                                   k=k,mu=mu_mag) + nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,Q,R_2,
                                                                            sigma=sigma_rm_2,k=k_2,mu=mu_2_mag)
    else:
        res = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,Q,R,sigma=sigma_rm,
                                   k=k,mu=mu_mag) + nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,Q,R_2,
                                                                            sigma=sigma_rm_2,k=k_2,mu=mu_2_mag)
    return res
"""
log variants for discrepancy joint fitting
"""
def log_single_intensity_spheroid(approx,
                              combinedfactor_log,
                              distribution,
                              formfactor,
                              volume,
                              Q,
                              R,  
                              gamma=4,
                              background_log=None,
                              C_log=None,
                              sigma=None,
                              k=None,
                              mu=None):
    combinedfactor = np.exp(-combinedfactor_log)
    
    sigma_rm = None
    if sigma is not None:
        sigma_rm = sigma*R
    if background_log is not None:
        background = np.exp(background_log)
        C = np.exp(C_log)
        res = final_intensity_spheroid(approx,combinedfactor,distribution,formfactor,volume,background,Q,R,C,gamma=gamma,sigma=sigma_rm,
                                   k=k,mu=mu)
    else:
        res =  nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,Q,R,sigma=sigma_rm,
                                   k=k,mu=mu)
    return np.log(res)
def log_double_intensity_spheroid(approx,
                              combinedfactor_log,
                              combinedfactor_2_log,
                              distribution,
                              distribution_2,
                              formfactor,
                              formfactor_2,
                              volume,
                              volume_2,
                              Q,
                              R,
                              R_2,
                              gamma=4,
                              background_log=None,
                              C_log=None,
                              sigma=None,
                              sigma_2=None,
                              k=None,
                              k_2=None,
                              mu=None,
                              mu_2=None):
    combinedfactor = np.exp(-combinedfactor_log)
    combinedfactor_2 = np.exp(-combinedfactor_2_log)
    sigma_rm = None
    sigma_rm_2 = None
    if sigma is not None:
        sigma_rm = sigma*R
    if sigma_2 is not None:
        sigma_rm_2 = sigma_2*R_2
    if background_log is not None:
        background = np.exp(background_log)
        C = np.exp(C_log)
        res = final_intensity_spheroid(approx,combinedfactor,distribution,formfactor,volume,background,Q,R,C,gamma=gamma,sigma=sigma_rm,
                                    k=k,mu=mu) + nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,Q,R_2,
                                                                                sigma=sigma_rm_2,k=k_2,mu=mu_2)
    else:
        res = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,Q,R,sigma=sigma_rm,
                                    k=k,mu=mu) + nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,Q,R_2,
                                                                                sigma=sigma_rm_2,k=k_2,mu=mu_2)
    return np.log(res)
"""
mag specific functions such that parameter names are different(very important)
"""
def log_single_intensity_spheroid_mag(approx,
                              combinedfactor_log_mag,
                              distribution,
                              formfactor,
                              volume,
                              Q,
                              R,
                              gamma=4,
                              background_log_mag=None,
                              C_log_mag=None,
                              sigma=None,
                              k=None,
                              mu_mag=None):
    combinedfactor = np.exp(-combinedfactor_log_mag)
    
    sigma_rm = None
    if sigma is not None:
        sigma_rm = sigma*R
    if background_log_mag is not None:
        background = np.exp(background_log_mag)
        C = np.exp(C_log_mag)
        res = final_intensity_spheroid(approx,combinedfactor,distribution,formfactor,volume,background,Q,R,C,gamma=gamma,sigma=sigma_rm,
                                   k=k,mu=mu_mag)
    else:
        res = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,Q,R,sigma=sigma_rm,
                                   k=k,mu=mu_mag)
    return np.log(res)

def log_double_intensity_spheroid_mag(approx,
                              combinedfactor_log_mag,
                              combinedfactor_2_log_mag,
                              distribution,
                              distribution_2,
                              formfactor,
                              formfactor_2,
                              volume,
                              volume_2,
                              
                              Q,
                              R,
                              R_2,
                              
                              gamma=4,
                              background_log_mag=None,
                              C_log_mag=None,
                              sigma=None,
                              sigma_2=None,
                              k=None,
                              k_2=None,
                              mu_mag=None,
                              mu_2_mag=None):
    combinedfactor = np.exp(-combinedfactor_log_mag)
    combinedfactor_2 = np.exp(-combinedfactor_2_log_mag)
    
    sigma_rm = None
    sigma_rm_2 = None
    if sigma is not None:
        sigma_rm = sigma*R
    if sigma_2 is not None:
        sigma_rm_2 = sigma_2*R_2
    if background_log_mag is not None:
        background = np.exp(background_log_mag)
        C = np.exp(C_log_mag)
        res = final_intensity_spheroid(approx,combinedfactor,distribution,formfactor,volume,background,Q,R,C,gamma=gamma,sigma=sigma_rm,
                                   k=k,mu=mu_mag) + nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,Q,R_2,
                                                                            sigma=sigma_rm_2,k=k_2,mu=mu_2_mag)
    else:
        res = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,Q,R,sigma=sigma_rm,
                                   k=k,mu=mu_mag) + nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,Q,R_2,
                                                                            sigma=sigma_rm_2,k=k_2,mu=mu_2_mag)
    return np.log(res)