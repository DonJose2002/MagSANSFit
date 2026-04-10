import numpy as np
import torch
from core.models.assembled_problem import final_intensity_spheroid
import inspect
#inspect if target chi2_red is reached
class TargetChi2Reached(Exception):
    def __init__(self, params):
        self.params = params

"""
Loss_chi2: Function for calculating chi squared loss.
experiment: input numpy array of experimental measurements at different Q
uncertainty: associated uncertainty
model: associated predicted values
num_parameters: number of parameters 
"""
def loss_chi2 (experiment, uncertainty, model, num_parameters):
    num_samples = np.size(experiment)
    return 1/(num_samples-num_parameters)*np.sum(((experiment-model)/uncertainty)**2)

#torch counterpart for chi2_red computation
def loss_chi2_torch(experiment, uncertainty, model, num_parameters):
    num_samples = experiment.shape[0]
    return 1/(num_samples-num_parameters)*torch.sum(((experiment-model)/uncertainty)**2)

"""
make_residual: function for constructing residual function, used in optimization.
func: model function
experiment: input numpy array of experimental measurements at different Q
uncertainty: associated uncertainty
param_names: list of optimizable numerical parameters
fixed_kwargs: parameters considered fixed during optimization
"""
def make_residual_nostop (func, x_array, x_name, experiment, uncertainty, param_names, fixed_kwargs):
    # --- auto-validate against the real signature ---
    sig = inspect.signature(func)
    all_args = set(sig.parameters.keys())
    
    provided = set(param_names) | set(fixed_kwargs.keys()) | {x_name}
    unknown  = provided - all_args

    # ignore parameters that have defaults and weren't provided
    required = {
        name for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
    }
    missing_required = required - provided

    if missing_required:
        raise ValueError(f"Required arguments not provided: {missing_required}")
    if unknown:
        raise ValueError(f"Unrecognised argument names: {unknown}")
    overlap = set(param_names) & set(fixed_kwargs.keys())
    if overlap:
        raise ValueError(f"Arguments appear in both param_names and fixed_kwargs: {overlap}")
    def residuals(params):
        optimized = dict(zip(param_names, params))
        shared_kwargs = {**fixed_kwargs, **optimized}
        y_model = np.array([func(**{**shared_kwargs, x_name: xi}) for xi in x_array])
        return (experiment - y_model) / uncertainty
    return residuals
"""
alternative version of make_residual with a stop criteria
"""
def make_residual(func, x_array, x_name, experiment, uncertainty, param_names, fixed_kwargs, target_chi2_red=1.0, tolerance=0.1):
    # --- auto-validate against the real signature ---
    sig = inspect.signature(func)
    all_args = set(sig.parameters.keys())
    
    provided = set(param_names) | set(fixed_kwargs.keys()) | {x_name}
    unknown  = provided - all_args

    # ignore parameters that have defaults and weren't provided
    required = {
        name for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
    }
    missing_required = required - provided

    if missing_required:
        raise ValueError(f"Required arguments not provided: {missing_required}")
    if unknown:
        raise ValueError(f"Unrecognised argument names: {unknown}")
    overlap = set(param_names) & set(fixed_kwargs.keys())
    if overlap:
        raise ValueError(f"Arguments appear in both param_names and fixed_kwargs: {overlap}")
    dof = len(experiment) - len(param_names)
    def residuals(params):
        optimized = dict(zip(param_names, params))
        shared_kwargs = {**fixed_kwargs, **optimized}
        y_model = np.array([func(**{**shared_kwargs, x_name: xi}) for xi in x_array])
        r = (experiment - y_model) / uncertainty
        chi2_red = np.sum(r**2) / dof

        if abs(chi2_red - target_chi2_red) < tolerance:
            raise TargetChi2Reached(params.copy())
        return (experiment - y_model) / uncertainty
    
    return residuals
    