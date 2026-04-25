from scipy.optimize import minimize
import numpy as np
def LBFGS_optimize(loss,startpoints,bounds):
    list_result = []
    for x0 in startpoints:
        result = minimize(loss, x0=x0, method='L-BFGS-B',
                                   bounds=bounds, options={
        "maxiter": 500,
        "ftol": 1e-9,
        "gtol": 1e-6
    })
        list_result.append(result.x)
    
    return list_result