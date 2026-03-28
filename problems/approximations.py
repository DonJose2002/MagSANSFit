"""
provides utils for guinier, porod, etc. approximations
"""
def I_porod(Q,gamma,C):
    return C/(Q**gamma)

