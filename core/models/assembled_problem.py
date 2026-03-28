from scipy.integrate import quad
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
            numerator,_ = quad(integrand_numerator,0.001,50)
            integrand_denominator = lambda Re: volume(Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50)
            I_nano = combinedfactor*numerator/denominator
            return I_nano
        elif mu is None: #ellipsoid
            integrand_numerator = lambda Re: formfactor(Q,Re,k*Re)**2*volume(Re,k*Re)**2*distribution(Re,R,sigma)
            numerator,_ = quad(integrand_numerator,0.001,50)
            integrand_denominator = lambda Re: volume(Re,k*Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50)
            I_nano = combinedfactor*numerator/denominator
            return I_nano
        else: #core-shell
            integrand_numerator = lambda Re: formfactor(Q,Re,k,mu)**2*volume(Re)**2*distribution(Re,R,sigma)
            numerator,_ = quad(integrand_numerator,0.001,50)
            integrand_denominator = lambda Re: volume(Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50)
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
gamma: fractal dimension of approximation function. This value still needs to be entered even with monodistribution, due to python syntax
"""

def final_intensity_spheroid(approx,combinedfactor,distribution,formfactor,volume,background,Q,R,C,gamma=4,sigma=None,k=None,mu=None):
    I_porod = approx(Q,gamma,C)
    if callable(distribution): #distribution with density
        if k is None: #sphere
            integrand_numerator = lambda Re: formfactor(Q,Re)**2*volume(Re)**2*distribution(Re,R,sigma)
            numerator,_ = quad(integrand_numerator,0.001,50)
            integrand_denominator = lambda Re: volume(Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50)
            I_nano = combinedfactor*numerator/denominator
            return I_porod + I_nano + background
        elif mu is None: #ellipsoid
            integrand_numerator = lambda Re: formfactor(Q,Re,k*Re)**2*volume(Re,k*Re)**2*distribution(Re,R,sigma)
            numerator,_ = quad(integrand_numerator,0.001,50)
            integrand_denominator = lambda Re: volume(Re,k*Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50)
            I_nano = combinedfactor*numerator/denominator
            return I_porod + I_nano + background
        else: #core-shell
            integrand_numerator = lambda Re: formfactor(Q,Re,k,mu)**2*volume(Re)**2*distribution(Re,R,sigma)
            numerator,_ = quad(integrand_numerator,0.001,50)
            integrand_denominator = lambda Re: volume(Re)*distribution(Re,R,sigma)
            denominator,_ = quad(integrand_denominator,0.001,50)
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

