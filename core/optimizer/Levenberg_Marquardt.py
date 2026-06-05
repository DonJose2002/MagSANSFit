from scipy.optimize import least_squares, OptimizeResult
from scipy.optimize import approx_fprime
import numpy as np

from core.models.loss_functions import make_residual, make_residual_nostop, TargetChi2Reached

"""
LM_optimize: optimizes target_function chi2 value based on the results of BO.
target_function: function whose chi2_red is to be optimized. Either single_intensity or double_intensity
x_array: array of points (Q) to evaluate target function
x_name: parameter name string of target function
experiment: experimental measurements
uncertainty: associate error bar
params: optimization target parameters
kwargs: fixed parameters
startpoints: initial guess parameters

result entries:
result.cost        # = 0.5 * sum(r²), so chi2 = result.cost * 2
result.x           # the best-fit parameters found
result.jac         # Jacobian matrix at the solution, shape (n_data, n_params)
result.status      # integer: >= 1 means success
result.message     # human-readable convergence message
"""
def LM_optimize(target_function,x_array,x_name,experiment,uncertainty,params,kwargs,startpoints,target_chi2_red_LM=1.0, tolerance_LM=0.1):
    residual = make_residual(target_function,x_array,x_name,experiment,uncertainty,params,kwargs,target_chi2_red=target_chi2_red_LM, tolerance=tolerance_LM)
    residual_nostop = make_residual_nostop(target_function,x_array,x_name,experiment,uncertainty,params,kwargs)
    best_result = None
    for x0 in startpoints:
        try:
            result = least_squares(residual,x0=x0,method='lm',ftol=1e-12, xtol=1e-12, gtol=1e-12)
        except TargetChi2Reached as e:
            result = least_squares(residual_nostop,x0=e.params,method='lm',ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=1)
        if best_result is None or result.cost < best_result.cost:
            best_result = result
    return best_result

def LM_optimize_nostop(target_function,x_array,x_name,experiment,uncertainty,params,kwargs,startpoints):
    residual_nostop = make_residual_nostop(target_function,x_array,x_name,experiment,uncertainty,params,kwargs)
    best_result = None
    for x0 in startpoints:
        result = least_squares(residual_nostop,x0=x0,method='lm',ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best_result is None or result.cost < best_result.cost:
            best_result = result
    return best_result

def LM_joint_optimize(joint_residual,joint_residual_nostop,startpoints):
    best_result = None
    for x0 in startpoints:
        try:
            result = least_squares(joint_residual, x0=x0, method='lm',
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12)
        except TargetChi2Reached as e:
            result = least_squares(joint_residual_nostop, x0=e.params, method='lm',
                                   ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=1)
        
        if best_result is None or result.cost < best_result.cost:
            best_result = result
    
    return best_result