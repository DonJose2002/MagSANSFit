import numpy as np
def formfactor_sphere(Q,R): # basis for form factor computation
    x = Q*R
    bessel = np.sin(x)/x**2 - np.cos(x)/x #calculate spherical Bessel Function
    f = 3/x*bessel
    return f
def volume_sphere(R):
    return 4/3*np.pi*R**3
