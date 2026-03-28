from problems.sphere import formfactor_sphere
"""
This form factor calculation does not give the form factor
for the core-shell model, rather an equivalent that can be
plugged directly into subsequent problems. Thus, the volume
of the core-shell is supposed the same as the sphere case.
"""
def formfactor_coreshell(Q,R,nu,mu):
    """
    mu: the relative contrast factor of the core:
    delta rho core = mu delta rho
    nu: the relative radius of the core:
    R core = nu R 
    """
    sphere_contribution = formfactor_sphere(Q,R)
    hole_contribution = (mu-1)*nu**3*formfactor_sphere(Q,nu*R)
    return sphere_contribution+hole_contribution
