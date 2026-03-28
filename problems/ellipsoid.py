import numpy as np
from scipy.integrate import quad
from problems.sphere import formfactor_sphere
"""
Auxiliary functions for form factor computation. 
Creates a composite function which is then integrated.
integration is done during execution of main.py and not here.
"""
def equivalent_radius_ab(a,b,x): # case of elongated ellipsoid, where a is its length in the direction of the axis of symmetry
    return a*np.sqrt(1+x**2*((b/a)**2-1))

def formfactor_ellipsoid_integrand(Q,a,b,x):
    return formfactor_sphere(Q,equivalent_radius_ab(a,b,x))

def formfactor_ellipsoid(Q,a,b):
    integrand = lambda x: formfactor_ellipsoid_integrand(Q,a,b,x)
    result, _ = quad(integrand,0,1)
    return result

def volume_ellipsoid(a,b):
    return 4/3*np.pi*a*b**2