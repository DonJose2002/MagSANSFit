import torch
import numpy as np
from core.models.loss_functions import loss_chi2_torch,loss_chi2

"""
objective: a transform for reduced chi2, for avoiding overfitting
chi2_red: calculated reduced chi2 values, torch tensor
target: optimal value for reduced chi2, usually 1
width: transformation steepness, larger width=larger tolerance
"""

def objective(chi2_red,target,width):
    reward = torch.exp(-0.5 * ((chi2_red - target) / width) ** 2)
    return reward

def objective_np(chi2_red,target,width):
    reward = np.exp(-0.5 * ((chi2_red - target) / width) ** 2)
    return reward
"""
reward_variance: the observation noise related with function objective. Model is deterministic, but experiment is not
"""
def reward_variance(experiment,uncertainty,model,num_parameters,chi2_red,target,width):
    residuals = experiment-model
    reward = objective_np(chi2_red,target,width)
     # Derivative of reward w.r.t. chi2_red
    d_reward_d_chi2red = -(chi2_red - target) / (width ** 2) * reward
    
    # Derivative of chi2_red w.r.t. each y_obs,i
    d_chi2red_d_yobs = -2 * residuals / (uncertainty ** 2 * (np.size(experiment)-num_parameters))  # shape: (n_obs,)
    
    # Chain rule: d_reward/d_yobs,i
    d_reward_d_yobs = d_reward_d_chi2red * d_chi2red_d_yobs     # shape: (n_obs,)
    
    # Variance by linear propagation: sum of (derivative * sigma)^2
    var_reward = np.sum((d_reward_d_yobs * uncertainty) ** 2)
    
    return var_reward

def chi2_red_variance(chi2_red, dof):
    return 4*chi2_red/dof

def log_chi2_red_variance(chi2_red, dof, s2=None): #s2: variance of log reduced chi2
    if s2 is not None:
        return torch.clamp(4/(dof*chi2_red), min=5e-6*s2)
    else:
        return torch.clamp(4/(dof*chi2_red), min=5e-6)
