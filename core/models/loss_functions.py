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

def noisy_loss_log_chi2 (experiment, uncertainty, model, discrepancy):
    log_y = np.log(experiment)
    log_f = np.log(model)
    propagated_uncertainty = uncertainty/experiment
    true_discrepancy = np.exp(discrepancy) #keep beta discrepancy in the log scale for consistency
    log_chi2 = np.sum(((log_y-log_f)**2)/(propagated_uncertainty**2+true_discrepancy**2))
    gaussian_normalization_term = np.sum(np.log(propagated_uncertainty**2+true_discrepancy**2))
    return log_chi2 + gaussian_normalization_term

def noisy_loss_log_chi2_separate (experiment, uncertainty, model, discrepancy):
    log_y = np.log(experiment)
    log_f = np.log(model)
    propagated_uncertainty = uncertainty/experiment
    true_discrepancy = np.exp(discrepancy) #keep beta discrepancy in the log scale for consistency
    log_chi2 = np.sum(((log_y-log_f)**2)/(propagated_uncertainty**2+true_discrepancy**2))
    gaussian_normalization_term = np.sum(np.log(propagated_uncertainty**2+true_discrepancy**2))
    return log_chi2, gaussian_normalization_term

def loss_log_chi2 (experiment,uncertainty,model):
    log_y = np.log(experiment)
    log_f = np.log(model)
    propagated_uncertainty = uncertainty/experiment
    return np.sum(((log_y-log_f)/propagated_uncertainty)**2)
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
    
"""
make_residuals_from_slice: function maker for nuc+mag joint fitting
"""
def make_residuals_from_slice(func,x_array,x_name, experiment, uncertainty, param_names, fixed_kwargs, all_param_names):
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
        local_params = np.array([
            params[all_param_names.index(name)]
            for name in param_names
        ])
        
        optimized = dict(zip(param_names, local_params))
        shared_kwargs = {**fixed_kwargs, **optimized}
        y_model = np.array([
            func(**{**shared_kwargs, x_name: xi})
            for xi in x_array
        ])
        return (experiment-y_model)/uncertainty
    return residuals
def make_discrepancy_loss_from_slice(func,x_array,x_name, experiment, uncertainty, param_names, fixed_kwargs, all_param_names, discrepancy):
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
        local_params = np.array([
            params[all_param_names.index(name)]
            for name in param_names
        ])
        
        optimized = dict(zip(param_names, local_params))
        shared_kwargs = {**fixed_kwargs, **optimized}
        y_model = np.array([
            func(**{**shared_kwargs, x_name: xi})
            for xi in x_array
        ])
        model_discrepancy = np.exp(params[all_param_names.index(discrepancy)])
        return np.sum((np.log(experiment)-np.log(y_model))**2/((uncertainty/y_model)**2+model_discrepancy**2) + np.log((uncertainty/y_model)**2+model_discrepancy**2))
    return residuals
"""
make_joint_residuals: creates single residual function
"""
def make_joint_residuals(residuals_1, residuals_2, dof = None, target_chi2_red = 1.0, tolerance = 0.2):
    def residuals(params):
        r1 = residuals_1(params)
        r2 = residuals_2(params)
        r = np.concatenate([r1,r2])
        if dof is not None:
            chi2_red_1 = np.sum(r1**2)/dof
            chi2_red_2 = np.sum(r2**2)/dof
            if (abs(chi2_red_1 - target_chi2_red) < tolerance) & (abs(chi2_red_2 - target_chi2_red) < tolerance):
                raise TargetChi2Reached(params.copy())
        return r
    return residuals
"""
make_weighted_joint_residuals: creates single weighted residual function based on input lambda coefficient.
used for pareto sweep
"""
def make_weighted_joint_residuals(residuals_1, residuals_2, lambda_coeff, dof_1, dof_2, target_chi2_red = 1.0, tolerance = 0.2, stop = True):
    def residuals(params):
        r1 = residuals_1(params)
        w1 = np.sqrt(lambda_coeff/dof_1)
        w2 = np.sqrt((1-lambda_coeff)/dof_2)
        r2 = residuals_2(params)
        r = np.concatenate([w1*r1,w2*r2])
        if stop:
            chi2_red_1 = np.sum(r1**2)/dof_1
            chi2_red_2 = np.sum(r2**2)/dof_2
            if (abs(chi2_red_1 - target_chi2_red) < tolerance) & (abs(chi2_red_2 - target_chi2_red) < tolerance):
                raise TargetChi2Reached(params.copy())
        return r
    return residuals

def make_discrepancy_loss(loss1,loss2):
    def discrepancy_loss(params):
        loss_nuc = loss1(params)
        loss_mag = loss2(params)
        res = loss_nuc+loss_mag
        return res
    return discrepancy_loss

def make_equivalent_chi2_loss_from_slice(func,x_array,x_name, experiment, uncertainty, param_names, fixed_kwargs, all_param_names, discrepancy):
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
        local_params = np.array([
            params[all_param_names.index(name)]
            for name in param_names
        ])
        
        optimized = dict(zip(param_names, local_params))
        shared_kwargs = {**fixed_kwargs, **optimized}
        y_model = np.array([
            func(**{**shared_kwargs, x_name: xi})
            for xi in x_array
        ])
        model_discrepancy = np.exp(params[all_param_names.index(discrepancy)])
        return np.sum((np.log(experiment)-np.log(y_model))**2/((uncertainty/y_model)**2+model_discrepancy**2))
    return residuals