import torch
import numpy as np
from core.models.loss_functions import loss_chi2_torch,loss_chi2
import copy
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
"""
Y Variance of reduced chi2, for logNEI of BO 
"""
def chi2_red_variance(chi2_red, dof):
    return 4*chi2_red/dof
"""
Y variance of log reduced chi2
"""
def log_chi2_red_variance(chi2_red, dof, s2=None): #s2: variance of log reduced chi2
    if s2 is not None:
        return torch.clamp(4/(dof*chi2_red), min=2e-6*s2) #rescale variance to suit computation limits
    else:
        return torch.clamp(4/(dof*chi2_red), min=2e-6)
"""
joint fit Y variance for log reduced chi2
"""
def joint_log_chi2_red_variance(chi2_red_1, chi2_red_2, dof, s2 = None):
    chi2_sum = chi2_red_1 + chi2_red_2
    variance_1 = 4*chi2_red_1/dof
    variance_2 = 4*chi2_red_2/dof
    variance = (variance_1+variance_2)/chi2_sum ** 2
    if s2 is not None:
        return torch.clamp(variance, min=s2*2e-6)
    else:
        return torch.clamp(variance, min=2e-6)
"""
compute_sigma_z: determines relative variance of experimental measurements
"""
def compute_sigma_z(experiment, uncertainty):
    return uncertainty/experiment
"""
construct_joint_bounds: returns a tensor for BO optimization based on 
"""
def construct_joint_bounds(jointstart,jointstart_mag,joint_searchwidth,pos_array_A_2,pos_array_mu_1,pos_array_mu_2,log_Ibg_mag,distribution_type,model_1,model_2):
    bounds_jointfit = [[jointstart[0]-joint_searchwidth[0],jointstart[0]+joint_searchwidth[0]],
                   [jointstart[1]-joint_searchwidth[1],jointstart[1]+joint_searchwidth[1]],
                   [jointstart[2]-joint_searchwidth[2],jointstart[2]+joint_searchwidth[2]]]


    for i in range(3, jointstart.size):
        if i != pos_array_A_2 and i != pos_array_mu_1 and i != pos_array_mu_2:
            if log_Ibg_mag is not None:
                i_mag = i
            else:
                i_mag = i-2 #decrement by 2 to match position
            current_bound = [max(jointstart[i]-joint_searchwidth[i],jointstart_mag[i_mag]-joint_searchwidth[i]),min(jointstart[i]+joint_searchwidth[i],jointstart_mag[i_mag]+joint_searchwidth[i])]
        else:
            current_bound = [jointstart[i]-joint_searchwidth[i],jointstart[i]+joint_searchwidth[i]]
        
        bounds_jointfit.append(current_bound)

    #add mag specific bounds
    if log_Ibg_mag is not None:
        decrement = 0
        bounds_mag_joint = [[jointstart_mag[0]-joint_searchwidth[0],jointstart_mag[0]+joint_searchwidth[0]],
                    [jointstart_mag[1]-joint_searchwidth[1],jointstart_mag[1]+joint_searchwidth[1]],
                    [jointstart_mag[2]-joint_searchwidth[2],jointstart_mag[2]+joint_searchwidth[2]]]
    else:
        decrement = 2
        bounds_mag_joint = [[jointstart_mag[0]-joint_searchwidth[2],jointstart_mag[0]+joint_searchwidth[2]]]

    if distribution_type == "double":
        bounds_mag_joint.append([jointstart_mag[pos_array_A_2-decrement]-joint_searchwidth[pos_array_A_2],jointstart_mag[pos_array_A_2-decrement]+joint_searchwidth[pos_array_A_2]])
        if model_1 == "core-shell":
            bounds_mag_joint.append([jointstart_mag[pos_array_mu_1-decrement]-joint_searchwidth[pos_array_mu_1],jointstart_mag[pos_array_mu_1-decrement]+joint_searchwidth[pos_array_mu_1]])
        if model_2 == "core-shell":
            bounds_mag_joint.append([jointstart_mag[pos_array_mu_2-decrement]-joint_searchwidth[pos_array_mu_2],jointstart_mag[pos_array_mu_2-decrement]+joint_searchwidth[pos_array_mu_2]])
    else:
        if model_1 == "core-shell":
            bounds_mag_joint.append([jointstart_mag[pos_array_mu_1-decrement]-joint_searchwidth[pos_array_mu_1],jointstart_mag[pos_array_mu_1-decrement]+joint_searchwidth[pos_array_mu_1]])

    bounds_final = torch.tensor(bounds_jointfit+bounds_mag_joint)
    bounds_final = bounds_final.T
    return bounds_final
"""
separate_nuc_mag_parameters: for reconstructing fit parameters after joint fit.
theta must be a numpy array
"""
def separate_nuc_mag_parameters(theta,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag):
    nuc_parameters = theta[:num_parameters]
    mag_parameters = nuc_parameters.copy()
    mag_parameters[nuc_pos_list] = theta[num_parameters+np.array(mag_pos_list)]
    if log_Ibg_mag is None:
        mag_parameters = mag_parameters[2:]
    return nuc_parameters, mag_parameters
"""
Alternative function call for torch.tensor
"""
def separate_nuc_mag_parameters_tensor(theta,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag):
    nuc_parameters = theta[:,:num_parameters]
    mag_parameters = nuc_parameters.clone()
    mag_parameters[:,nuc_pos_list] = theta[:,num_parameters+torch.tensor(mag_pos_list)]
    if log_Ibg_mag is None:
        mag_parameters = mag_parameters[:,2:]
    return nuc_parameters, mag_parameters
"""
construct_joint_parameters: reverse operation of separate_nuc_mag_parameters. 
Likewise, nuc_parameters and mag_parameters must be numpy arrays
"""
def construct_joint_parameters(nuc_parameters,mag_parameters,pos_array_A_2,pos_array_mu_1,pos_array_mu_2,log_Ibg_mag,distribution_type,model_1,model_2):
    theta = nuc_parameters
    if log_Ibg_mag is not None:
        decrement = 0
        theta_mag = [mag_parameters[0],mag_parameters[1],mag_parameters[2]]
    else:
        decrement = 2
        theta_mag = [mag_parameters[0]]

    if distribution_type == "double":
        theta_mag.append(mag_parameters[pos_array_A_2-decrement])
        if model_1 == "core-shell":
            theta_mag.append(mag_parameters[pos_array_mu_1-decrement])
        if model_2 == "core-shell":
            theta_mag.append(mag_parameters[pos_array_mu_2-decrement])
    else:
        if model_1 == "core-shell":
            theta_mag.append(mag_parameters[pos_array_mu_1-decrement])
    theta_mag_np = np.array(theta_mag)
    theta_np = np.concatenate((theta,theta_mag_np))
    return theta_np
"""
assemble_theta: used to gather theta information from nuc_params, mag_params and common params
joint_params: list of all param names to be gathered
nuc_params_names: list of nuclear fitted parameters
nuc_param_values: corresponding values in numpy array
mag_param_names,mag_params_values: idem
common_param_dict: common params and their values, in dict format
"""
def assemble_theta(joint_param_names,nuc_param_names,nuc_param_values,mag_param_names,mag_param_values,common_param_dict):
    mapping = {}
    for name,value in zip(nuc_param_names,nuc_param_values):
        if name in mapping:
            raise ValueError(f"Duplicate key '{name}' found in nuc_param_names/nuc_param_values and previous mapping.")
        mapping[name] = value
    for name,value in zip(mag_param_names,mag_param_values):
        if name in mapping:
            raise ValueError(f"Duplicate key '{name}' found in mag_param_names/mag_param_values and previous mapping.")
        mapping[name] = value
    for name,value in common_param_dict.items():
        if name in mapping:
            raise ValueError(f"Duplicate key '{name}' found in common_param_dict and previous mapping.")
        mapping[name] = value
    joint_param_values = []
    for name in joint_param_names:
        if name in mapping:
            joint_param_values.append(mapping[name])
        else:
            raise ValueError(f"String '{name}' not found in nuc_param_names, mag_param_names, or common_param_dict.")
    joint_param_values_np = np.array(joint_param_values)
    return joint_param_values_np
