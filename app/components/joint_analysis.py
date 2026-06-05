import streamlit as st
import numpy as np
import torch
from botorch.acquisition.analytic import LogNoisyExpectedImprovement,LogExpectedImprovement
from botorch.optim import optimize_acqf
import os
import copy
import math

from app.utilities.cache_generator import get_model_config

from problems.approximations import I_porod
from core.utils.helper_functions import construct_joint_bounds,separate_nuc_mag_parameters,assemble_theta,separate_nuc_mag_parameters_tensor,joint_log_chi2_red_variance
from core.models.BO_model import chi2_final_intensity_spheroid_BO_single,chi2_final_intensity_spheroid_BO_double,chi2_nano_intensity_spheroid_BO_double,chi2_nano_intensity_spheroid_BO_single
from core.models.BO_with_error_separate import log_chi2_final_intensity_spheroid_BO_double_separate,log_chi2_final_intensity_spheroid_BO_single_separate,log_chi2_nano_intensity_spheroid_BO_double_separate,log_chi2_nano_intensity_spheroid_BO_single_separate
from core.models.BO_model_fixed import chi2_final_intensity_spheroid_BO_double_fixed,chi2_final_intensity_spheroid_BO_single_fixed,chi2_nano_intensity_spheroid_BO_double_fixed,chi2_nano_intensity_spheroid_BO_single_fixed
from core.models.loss_functions import make_joint_residuals,make_residuals_from_slice,make_discrepancy_loss_from_slice,make_discrepancy_loss,make_discrepancy_loss_from_slice_noresidual,make_weighted_joint_residuals
from core.models.single_loss import chi2_final_intensity_spheroid_double,chi2_final_intensity_spheroid_single,chi2_nano_intensity_spheroid_double,chi2_nano_intensity_spheroid_single
from core.optimizer.Levenberg_Marquardt import LM_optimize, LM_joint_optimize
from core.optimizer.LBFGS import LBFGS_optimize
from core.acquisition.acquisition_functions import build_model,build_model_discrepancy,build_model_nonoise
from core.models.assembled_problem import final_intensity_spheroid,nano_intensity_spheroid,double_intensity_spheroid,single_intensity_spheroid,double_intensity_spheroid_mag, single_intensity_spheroid_mag


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
def joint_analysis(treated_input,global_model_state,analysis_configs,single_fit_result,BO_config):
    ### gather st info ###
    analysis_mode =  global_model_state["analysis_mode"]
    distribution_type = global_model_state["distribution_type"]
    magnetic_scattering_background = global_model_state["magnetic_scattering_background"]
    model_1 = global_model_state["model_1"]
    model_2 = global_model_state["model_2"]
    model_state = get_model_config(global_model_state)
    ### input decode ###
    if distribution_type == "Double":
        distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2 = model_state
    elif distribution_type == "Single":
        distribution_func_1,formfactor_1,volume_1 = model_state
        
    test_Q, test_Inuc, test_sigmaInuc, test_sigmaQ, test_Imag, test_sigmaImag = treated_input
    sigma_z_nuc = np.average(test_sigmaInuc/test_Inuc)
    sigma_z_mag = np.average(test_sigmaImag/test_Imag)
    model_Q = np.linspace(max(np.min(test_Q)-0.1,0.01),np.max(test_Q)+0.1,100)
    if magnetic_scattering_background == "Off":
        log_Ibg_mag = None
    else:
        log_Ibg_mag = 0 # placeholder value for logic

    max_iteration_BO = analysis_configs["max_iteration_BO"]
    max_iteration_BO_Pareto = analysis_configs["max_iteration_BO_Pareto"]
    min_discrepancy_iteration = analysis_configs["min_discrepancy_iteration"]
    logspace_tolerance = analysis_configs["logspace_tolerance"]
    n_init = analysis_configs["n_init"]
    n_init_discrepancy = analysis_configs["n_init_discrepancy"]
    n_init_fine = analysis_configs["n_init_fine"]
    min_model_discrepancy = analysis_configs["min_model_discrepancy"]
    max_model_discrepancy = analysis_configs["max_model_discrepancy"]

    (bounds, 
            bounds_mag, 
            params,
            params_mag,
            common_params,
            num_parameters, 
            fixed_kwargs,
            params_mag_1, 
            params_firstfit_mag, 
            mask_params_nuc, 
            mask_params_mag, 
            possible_common_params_double, 
            possible_common_params_single,
            nuc_params_filtered,
            bounds_nuc_filtered,
            mag_params_filtered,
            bounds_mag_filtered,
            deleted_nuc_params,
            deleted_mag_params,
            joint_searchwidth,
            joint_searchwidth_fine,
            jointfit_params,
            nuc_pos_list,
            mag_pos_list,
            pos_array_A_2,
            pos_array_mu_1,
            pos_array_mu_2) = BO_config
    dof_nuc = test_Inuc.shape[0]-bounds.shape[1]
    dof_mag = test_Imag.shape[0]-bounds_mag.shape[1]
    jointstart_nuc, chi2_nuc = single_fit_result["Nuclear Signal"]
    jointstart_mag, chi2_mag = single_fit_result["Magnetic Signal"]
    #print(jointfit_params)
    ### first do simple joint fit ###
    bounds_jointfit = construct_joint_bounds(jointstart_nuc,jointstart_mag,joint_searchwidth,pos_array_A_2,pos_array_mu_1,pos_array_mu_2,log_Ibg_mag,distribution_type,model_1,model_2)
    print(bounds_jointfit)
    jointfit_X = bounds_jointfit[0] + (bounds_jointfit[1]-bounds_jointfit[0])*torch.rand(n_init,bounds_jointfit.shape[1])
    beta_jointfit = 3.0/(1/bounds_jointfit.shape[1]**0.5)
    #construct separate starter values
    jointfit_X_nuc, jointfit_X_mag = separate_nuc_mag_parameters_tensor(jointfit_X,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
    #initialize Y values
    if distribution_type == "Double":
        jointfit_Y_nuc = chi2_final_intensity_spheroid_BO_double(jointfit_X_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
        if log_Ibg_mag is not None:
            jointfit_Y_mag = chi2_final_intensity_spheroid_BO_double(jointfit_X_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            jointfit_Y_mag = chi2_nano_intensity_spheroid_BO_double(jointfit_X_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        jointfit_Y_nuc = chi2_final_intensity_spheroid_BO_single(jointfit_X_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
        if log_Ibg_mag is not None: 
            jointfit_Y_mag = chi2_final_intensity_spheroid_BO_single(jointfit_X_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        else:
            jointfit_Y_mag = chi2_nano_intensity_spheroid_BO_single(jointfit_X_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)

    jointfit_Y_nuc = jointfit_Y_nuc.unsqueeze(-1)
    jointfit_Y_mag = jointfit_Y_mag.unsqueeze(-1)
    jointfit_Y = jointfit_Y_nuc + jointfit_Y_mag 



    log_Y_jointfit = torch.log(jointfit_Y)

    #BO iteration
    for iteration in range(max_iteration_BO):
        s2_jointfit = log_Y_jointfit.var()
        jointfit_Y_variance = joint_log_chi2_red_variance(jointfit_Y_nuc,jointfit_Y_mag,dof_nuc,dof_mag,s2_jointfit)
        model_gp = build_model(jointfit_X,log_Y_jointfit,jointfit_Y_variance,beta_jointfit)
        
        acq = LogNoisyExpectedImprovement(model_gp,jointfit_X,maximize = False)
        candidate,_ = optimize_acqf(
            acq_function = acq,
            bounds = bounds_jointfit,
            q = 1,
            num_restarts = 20,
            raw_samples = 512,
        )
        new_jointfit_X_nuc = candidate[:,:num_parameters]
        new_jointfit_X_mag = new_jointfit_X_nuc.clone()
        new_jointfit_X_mag[:, nuc_pos_list] = candidate[:, num_parameters+torch.tensor(mag_pos_list)]
        if log_Ibg_mag is None:
            new_jointfit_X_mag = new_jointfit_X_mag[:,2:]
        if distribution_type == "Double":
            new_jointfit_Y_nuc = chi2_final_intensity_spheroid_BO_double(new_jointfit_X_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None:
                new_jointfit_Y_mag = chi2_final_intensity_spheroid_BO_double(new_jointfit_X_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                new_jointfit_Y_mag = chi2_nano_intensity_spheroid_BO_double(new_jointfit_X_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            new_jointfit_Y_nuc = chi2_final_intensity_spheroid_BO_single(new_jointfit_X_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None: 
                new_jointfit_Y_mag = chi2_final_intensity_spheroid_BO_single(new_jointfit_X_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            else:
                new_jointfit_Y_mag = chi2_nano_intensity_spheroid_BO_single(new_jointfit_X_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        new_jointfit_Y_nuc = new_jointfit_Y_nuc.unsqueeze(-1)
        new_jointfit_Y_mag = new_jointfit_Y_mag.unsqueeze(-1)
        new_jointfit_Y = new_jointfit_Y_nuc + new_jointfit_Y_mag
        new_log_Y_jointfit = torch.log(new_jointfit_Y)
        #new_jointfit_Y_variance = joint_log_chi2_red_variance(new_jointfit_Y_nuc,new_jointfit_Y_mag,dof)
        jointfit_X = torch.cat([jointfit_X, candidate])
        jointfit_Y_nuc = torch.cat([jointfit_Y_nuc, new_jointfit_Y_nuc])
        jointfit_Y_mag = torch.cat([jointfit_Y_mag, new_jointfit_Y_mag])
        jointfit_Y = torch.cat([jointfit_Y, new_jointfit_Y])
        log_Y_jointfit = torch.cat([log_Y_jointfit, new_log_Y_jointfit])
        #jointfit_Y_variance = torch.cat([jointfit_Y_variance,new_jointfit_Y_variance])
        print(f"Iteration {iteration}: best target = {jointfit_Y.min().item():.4f}, new candidate = {candidate}")

        condition_jointfit = (jointfit_Y_nuc <= 2.0) & (jointfit_Y_mag <= 2.0)
        count = torch.sum(condition_jointfit).item()
        if count >= 5:
            #indices = torch.nonzero(condition)
            #print(f"Acceptable fit: chi2_red = {Y.min().item():.3f}")
            break

    best_idx_jointfit = torch.argmin(jointfit_Y)
    print("Best theta:", jointfit_X[best_idx_jointfit])
    print("Best combined chi2:", jointfit_Y[best_idx_jointfit])
    if torch.any(condition_jointfit).item():
        condition_jointfit = condition_jointfit.squeeze()
        #indices_jointfit = torch.nonzero(condition_jointfit)
        startpoints_jointfit = jointfit_X[condition_jointfit]
        if count != 1:
            BO_candidates_jointfit = startpoints_jointfit.detach().cpu().numpy()
        else:
            BO_candidates_jointfit = [jointfit_X[best_idx_jointfit].detach().cpu().numpy()]
    else:
        BO_candidates_jointfit = [jointfit_X[best_idx_jointfit].detach().cpu().numpy()]
        #BO_candidates_jointfit.reshape(1,bounds_final.shape[1])


    if distribution_type == "Double":
        residual_nuc = make_residuals_from_slice(double_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params,fixed_kwargs,jointfit_params)
        residual_mag = make_residuals_from_slice(double_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
        joint_residual_nostop = make_joint_residuals(residual_nuc,residual_mag)
        joint_residual = make_joint_residuals(residual_nuc,residual_mag,dof=dof_nuc,dof_2=dof_mag)
        result_jointfit_1 = LM_joint_optimize(joint_residual,joint_residual_nostop,startpoints=BO_candidates_jointfit)
    else:
        residual_nuc = make_residuals_from_slice(single_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params,fixed_kwargs,jointfit_params)
        residual_mag = make_residuals_from_slice(single_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
        joint_residual_nostop = make_joint_residuals(residual_nuc,residual_mag)
        joint_residual = make_joint_residuals(residual_nuc,residual_mag,dof=dof_nuc,dof_2=dof_mag)
        result_jointfit_1 = LM_joint_optimize(joint_residual,joint_residual_nostop,startpoints=BO_candidates_jointfit)

    J_jointfit_1 = result_jointfit_1.jac #jacobian
    H_approx_jointfit_1 = J_jointfit_1.T @ J_jointfit_1 #hessian approximation
    cov_jointfit_1 = np.linalg.inv(H_approx_jointfit_1) #covariance matrix
    uncertainties_jointfit_1 = np.sqrt(np.diag(cov_jointfit_1)) #fit uncertainties

    print(f"Best result: Parameters = {result_jointfit_1.x}, chi2 = {result_jointfit_1.cost*2:.4f}")
    theta_jointfit_1 = result_jointfit_1.x
    ### Analyze first jointfit result ###
    theta_jointfit_1_nuc, theta_jointfit_1_mag = separate_nuc_mag_parameters(theta_jointfit_1, num_parameters, nuc_pos_list, mag_pos_list, log_Ibg_mag)
    if distribution_type == "Double":
        chi2_jointfit_1_nuc = chi2_final_intensity_spheroid_double(theta_jointfit_1_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
        if log_Ibg_mag is not None:
            chi2_jointfit_1_mag = chi2_final_intensity_spheroid_double(theta_jointfit_1_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            chi2_jointfit_1_mag = chi2_nano_intensity_spheroid_double(theta_jointfit_1_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        chi2_jointfit_1_nuc = chi2_final_intensity_spheroid_single(theta_jointfit_1_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
        if log_Ibg_mag is not None:
            chi2_jointfit_1_mag = chi2_final_intensity_spheroid_single(theta_jointfit_1_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        else:
            chi2_jointfit_1_mag = chi2_nano_intensity_spheroid_single(theta_jointfit_1_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    print(chi2_jointfit_1_nuc,chi2_jointfit_1_mag)
    if chi2_jointfit_1_nuc-1 < 0.2 and chi2_jointfit_1_mag-1 < 0.2:
        print(f"First joint fit results satisfactory: chi2_nuc = {chi2_jointfit_1_nuc:.3f}, chi2_mag = {chi2_jointfit_1_mag:.3f}. Returning result as output file.")
        lookup = dict(zip(jointfit_params, theta_jointfit_1))
        log_Ibg = lookup.get("background_log")
        log_C = lookup.get("C_log")
        A = lookup.get("combinedfactor_log")
        Rm_1 = lookup.get("R")
        sigma_1 = lookup.get("sigma")
        k_1 = lookup.get("k")
        mu_1 = lookup.get("mu")
        A_2 = lookup.get("combinedfactor_2_log")
        Rm_2 = lookup.get("R_2")
        sigma_2 = lookup.get("sigma_2")
        k_2 = lookup.get("k_2")
        mu_2 = lookup.get("mu_2")
        log_Ibg_mag = lookup.get("background_log_mag")
        log_C_mag = lookup.get("C_log_mag")
        A_mag = lookup.get("combinedfactor_log_mag")
        A_2_mag = lookup.get("combinedfactor_2_log_mag")
        mu_1_mag = lookup.get("mu_mag")
        mu_2_mag = lookup.get("mu_2_mag")
        combinedfactor_1 = np.exp(-A)
        combinedfactor_1_mag = np.exp(-A_mag)
        if log_Ibg is not None:
            Ibg = np.exp(log_Ibg)
            C = np.exp(log_C)
        if sigma_1 is not None:
            sigma_rm_1 = sigma_1*Rm_1
        if sigma_2 is not None:
            sigma_rm_2 = sigma_2*Rm_2
        if A_2 is not None:
            combinedfactor_2 = np.exp(-A_2)
        if log_Ibg_mag is not None:
            Ibg_mag = np.exp(log_Ibg_mag)
            C_mag = np.exp(log_C_mag)
        if A_2_mag is not None:
            combinedfactor_2_mag = np.exp(-A_2_mag)
        
        if distribution_type == "Double":
            model_Inuc = [(final_intensity_spheroid(
                        I_porod,combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                        Ibg,q,Rm_1,C,sigma=sigma_rm_1,k=k_1,mu=mu_1
                        )+nano_intensity_spheroid(combinedfactor_2,distribution_func_2,formfactor_2,volume_2,
                                                q,Rm_2,sigma=sigma_rm_2,k=k_2,mu=mu_2)) for q in model_Q]
            if log_Ibg_mag is not None:
                model_Imag = [(final_intensity_spheroid(
                        I_porod,combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                        Ibg_mag,q,Rm_1,C_mag,sigma=sigma_rm_1,k=k_1,mu=mu_1_mag
                        )+nano_intensity_spheroid(combinedfactor_2_mag,distribution_func_2,formfactor_2,volume_2,
                                                q,Rm_2,sigma=sigma_rm_2,k=k_2,mu=mu_2_mag)) for q in model_Q]
            else:
                model_Imag = [(nano_intensity_spheroid(
                        combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                        q,Rm_1,sigma=sigma_rm_1,k=k_1,mu=mu_1_mag
                        )+nano_intensity_spheroid(combinedfactor_2_mag,distribution_func_2,formfactor_2,volume_2,
                                                q,Rm_2,sigma=sigma_rm_2,k=k_2,mu=mu_2_mag)) for q in model_Q]
        elif distribution_type == "Single":
            model_Inuc = [final_intensity_spheroid(
                        I_porod,combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                        Ibg,q,Rm_1,C,sigma=sigma_rm_1,k=k_1,mu=mu_1
                        ) for q in model_Q]
            if log_Ibg_mag is not None:
                model_Imag = [final_intensity_spheroid(
                        I_porod,combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                        Ibg_mag,q,Rm_1,C_mag,sigma=sigma_rm_1,k=k_1,mu=mu_1_mag
                        ) for q in model_Q]
            else:                    
                model_Imag = [nano_intensity_spheroid(
                        combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                        q,Rm_1,sigma=sigma_rm_1,k=k_1,mu=mu_1_mag
                        ) for q in model_Q]

        original_file = st.session_state["data_loader"].name
        original_file_name = original_file.replace(".txt","")
        output_file_name = f"{original_file_name}_Joint_Analysis_firststage"
        output_file = output_file_name.replace(" ","_") + ".txt"
        filepath = os.path.join(
            OUTPUT_DIR,
            output_file
        )
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"#Original file: {original_file} \n")
            f.write(f"### Analysis Options: ### \n")
            f.write(f"#Distribution Type: {distribution_type} \n")
            f.write(f"#First distribution: {global_model_state["distribution_1"]} \n")
            f.write(f"#First model: {global_model_state["model_1"]} \n")
            if distribution_type == "Double":
                f.write(f"#Second distribution: {global_model_state["distribution_2"]} \n")
                f.write(f"#Second model: {global_model_state["model_2"]} \n")
            f.write(f"#Magnetic Scattering Background: {magnetic_scattering_background} \n")
            f.write(f"### Fit Satisfactory at stage 1: Simple Joint Fit ### \n")
            f.write(f"### Fit Results: ### \n")
            for idx in range(len(jointfit_params)):
                variable_name = jointfit_params[idx]
                variable_value = theta_jointfit_1[idx]
                variable_uncertainty = uncertainties_jointfit_1[idx]
                f.write(f"#{variable_name}: {variable_value}, uncertainty: {variable_uncertainty} \n")
            f.write(f"#Chi2_red_nuc: {chi2_jointfit_1_nuc} \n" )
            f.write(f"#Chi2_red_mag: {chi2_jointfit_1_mag} \n" )
            f.write(f"### Model Intensity ### \n")
            f.write(f"#Q(nm-1) Inuc(Q)(cm-1) Imag(Q)(cm-1)\n")
            for q, Inuc, Imag in zip(model_Q,model_Inuc,model_Imag):
                f.write(f"{q} {Inuc} {Imag}\n")

        print("Joint Analysis finished!")
        return theta_jointfit_1
    
    ### Second Stage: Joint fit with discrepancy term ###
    print("Simple joint fit results not satisfactory. Resorting to discrepancy analysis.")
    ###build discrepancy fit bounds from jointfit bounds
    bounds_jointfit_nucpart = [[jointstart_nuc[0]-joint_searchwidth[0],jointstart_nuc[0]+joint_searchwidth[0]],
                   [jointstart_nuc[1]-joint_searchwidth[1],jointstart_nuc[1]+joint_searchwidth[1]],
                   [jointstart_nuc[2]-joint_searchwidth[2],jointstart_nuc[2]+joint_searchwidth[2]]]


    for i in range(3, jointstart_nuc.size):
        if i != pos_array_A_2 and i != pos_array_mu_1 and i != pos_array_mu_2:
            if log_Ibg_mag is not None:
                i_mag = i
            else:
                i_mag = i-2 #decrement by 2 to match position
            current_bound = [min(jointstart_nuc[i]-joint_searchwidth[i],jointstart_mag[i_mag]-joint_searchwidth[i]),max(jointstart_nuc[i]+joint_searchwidth[i],jointstart_mag[i_mag]+joint_searchwidth[i])]
        else:
            current_bound = [jointstart_nuc[i]-joint_searchwidth[i],jointstart_nuc[i]+joint_searchwidth[i]]
        
        bounds_jointfit_nucpart.append(current_bound)
    #add mag specific bounds
    if log_Ibg_mag is not None:
        decrement = 0
        bounds_jointfit_magpart = [[jointstart_mag[0]-joint_searchwidth[0],jointstart_mag[0]+joint_searchwidth[0]],
                    [jointstart_mag[1]-joint_searchwidth[1],jointstart_mag[1]+joint_searchwidth[1]],
                    [jointstart_mag[2]-joint_searchwidth[2],jointstart_mag[2]+joint_searchwidth[2]]]
    else:
        decrement = 2
        bounds_jointfit_magpart = [[jointstart_mag[0]-joint_searchwidth[2],jointstart_mag[0]+joint_searchwidth[2]]]

    if distribution_type == "Double":
        bounds_jointfit_magpart.append([jointstart_mag[pos_array_A_2-decrement]-joint_searchwidth[pos_array_A_2],jointstart_mag[pos_array_A_2-decrement]+joint_searchwidth[pos_array_A_2]])
        if model_1 == "core-shell":
            bounds_jointfit_magpart.append([jointstart_mag[pos_array_mu_1-decrement]-joint_searchwidth[pos_array_mu_1],jointstart_mag[pos_array_mu_1-decrement]+joint_searchwidth[pos_array_mu_1]])
        if model_2 == "core-shell":
            bounds_jointfit_magpart.append([jointstart_mag[pos_array_mu_2-decrement]-joint_searchwidth[pos_array_mu_2],jointstart_mag[pos_array_mu_2-decrement]+joint_searchwidth[pos_array_mu_2]])
    else:
        if model_1 == "core-shell":
            bounds_jointfit_magpart.append([jointstart_mag[pos_array_mu_1-decrement]-joint_searchwidth[pos_array_mu_1],jointstart_mag[pos_array_mu_1-decrement]+joint_searchwidth[pos_array_mu_1]])
    range_model_discrepancy_log = [np.log(min_model_discrepancy), np.log(max_model_discrepancy)]
    bounds_discrepancy_joint = copy.deepcopy(bounds_jointfit_nucpart)
    bounds_discrepancy_joint.append(range_model_discrepancy_log)
    bounds_discrepancy_mag_joint = copy.deepcopy(bounds_jointfit_magpart)
    bounds_discrepancy_mag_joint.append(range_model_discrepancy_log)
    bounds_discrepancy_final_LBFGS = bounds_discrepancy_joint + bounds_discrepancy_mag_joint
    bounds_for_LBFGS = [tuple(pair) for pair in bounds_discrepancy_final_LBFGS]
    bounds_discrepancy_final = torch.tensor(bounds_discrepancy_joint+bounds_discrepancy_mag_joint)
    bounds_discrepancy_final = bounds_discrepancy_final.T

    ###inital guess values
    n_init_discrepancy = 100
    discrepancy_X = bounds_discrepancy_final[0] + (bounds_discrepancy_final[1]-bounds_discrepancy_final[0])*torch.rand(n_init_discrepancy,bounds_discrepancy_final.shape[1])
    #beta_discrepancy = 3.0/(1/bounds_discrepancy_final.shape[1]**0.5)
    beta_discrepancy = 6
    #construct separate starter values
    discrepancy_X_nuc = discrepancy_X[:,:num_parameters+1]

    discrepancy_X_mag = discrepancy_X_nuc.clone()

    discrepancy_X_mag[:, nuc_pos_list] = discrepancy_X[:, num_parameters+1+torch.tensor(mag_pos_list)]
    discrepancy_X_mag[:,-1] = discrepancy_X[:,-1]
    if log_Ibg_mag is None:
        discrepancy_X_mag = discrepancy_X_mag[:,2:]

    #initialize Y values
    if distribution_type == "Double":
        discrepancy_Y_nuc, discrepancy_Y_nuc_residual = log_chi2_final_intensity_spheroid_BO_double_separate(discrepancy_X_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
        if log_Ibg_mag is not None:
            discrepancy_Y_mag, discrepancy_Y_mag_residual = log_chi2_final_intensity_spheroid_BO_double_separate(discrepancy_X_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            discrepancy_Y_mag, discrepancy_Y_mag_residual = log_chi2_nano_intensity_spheroid_BO_double_separate(discrepancy_X_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        discrepancy_Y_nuc, discrepancy_Y_nuc_residual= log_chi2_final_intensity_spheroid_BO_single_separate(discrepancy_X_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
        if log_Ibg_mag is not None: 
            discrepancy_Y_mag, discrepancy_Y_mag_residual = log_chi2_final_intensity_spheroid_BO_single_separate(discrepancy_X_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        else:
            discrepancy_Y_mag, discrepancy_Y_mag_residual = log_chi2_nano_intensity_spheroid_BO_single_separate(discrepancy_X_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)

    discrepancy_Y_nuc = discrepancy_Y_nuc.unsqueeze(-1)
    discrepancy_Y_nuc_residual = discrepancy_Y_nuc_residual.unsqueeze(-1)
    discrepancy_Y_mag = discrepancy_Y_mag.unsqueeze(-1)
    discrepancy_Y_mag_residual = discrepancy_Y_mag_residual.unsqueeze(-1)
    discrepancy_Y = discrepancy_Y_nuc + discrepancy_Y_mag #corresponds to tampered chi2
    discrepancy_Y_residual = discrepancy_Y_nuc_residual + discrepancy_Y_mag_residual # gaussian normalization residual term
    discrepancy_Y_total = discrepancy_Y + discrepancy_Y_residual
    log_discrepancy_Y_total = torch.log(discrepancy_Y_total)
    log_discrepancy_Y_total_best = None
    acceptable_candidates = []
    count = 0
    print(dof_nuc,dof_mag)
    #BO iteration
    for iteration in range(max_iteration_BO):
        model_gp = build_model_discrepancy(discrepancy_X,log_discrepancy_Y_total,beta_discrepancy)
        best_f = discrepancy_Y_total.min().item()
        acq = LogExpectedImprovement(model_gp,best_f,maximize = False)
        candidate,_ = optimize_acqf(
            acq_function = acq,
            bounds = bounds_discrepancy_final,
            q = 1,
            num_restarts = 20,
            raw_samples = 512,
        )
        new_discrepancy_X_nuc = candidate[:,:num_parameters+1]
        new_discrepancy_X_mag = new_discrepancy_X_nuc.clone()
        new_discrepancy_X_mag[:, nuc_pos_list] = candidate[:, num_parameters+1+torch.tensor(mag_pos_list)]
        new_discrepancy_X_mag[:,-1] = candidate[:,-1]
        if log_Ibg_mag is None:
            new_discrepancy_X_mag = new_discrepancy_X_mag[:,2:]
        if distribution_type == "Double":
            new_discrepancy_Y_nuc, new_discrepancy_Y_nuc_residual = log_chi2_final_intensity_spheroid_BO_double_separate(new_discrepancy_X_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None:
                new_discrepancy_Y_mag, new_discrepancy_Y_mag_residual = log_chi2_final_intensity_spheroid_BO_double_separate(new_discrepancy_X_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                new_discrepancy_Y_mag, new_discrepancy_Y_mag_residual = log_chi2_nano_intensity_spheroid_BO_double_separate(new_discrepancy_X_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            new_discrepancy_Y_nuc, new_discrepancy_Y_nuc_residual = log_chi2_final_intensity_spheroid_BO_single_separate(new_discrepancy_X_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None: 
                new_discrepancy_Y_mag, new_discrepancy_Y_mag_residual = log_chi2_final_intensity_spheroid_BO_single_separate(new_discrepancy_X_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            else:
                new_discrepancy_Y_mag, new_discrepancy_Y_mag_residual = log_chi2_nano_intensity_spheroid_BO_single_separate(new_discrepancy_X_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        new_discrepancy_Y_nuc = new_discrepancy_Y_nuc.unsqueeze(-1)
        new_discrepancy_Y_nuc_residual = new_discrepancy_Y_nuc_residual.unsqueeze(-1)
        new_discrepancy_Y_mag = new_discrepancy_Y_mag.unsqueeze(-1)
        new_discrepancy_Y_mag_residual = new_discrepancy_Y_mag_residual.unsqueeze(-1)
        new_discrepancy_Y = new_discrepancy_Y_nuc + new_discrepancy_Y_mag 
        new_discrepancy_Y_residual = new_discrepancy_Y_nuc_residual + new_discrepancy_Y_mag_residual
        new_discrepancy_Y_total = new_discrepancy_Y + new_discrepancy_Y_residual
        new_log_discrepancy_Y_total = torch.log(new_discrepancy_Y_total)
        
        discrepancy_X = torch.cat([discrepancy_X, candidate])
        discrepancy_Y_nuc = torch.cat([discrepancy_Y_nuc, new_discrepancy_Y_nuc])
        discrepancy_Y_nuc_residual = torch.cat([discrepancy_Y_nuc_residual, new_discrepancy_Y_nuc_residual])
        discrepancy_Y_mag = torch.cat([discrepancy_Y_mag, new_discrepancy_Y_mag])
        discrepancy_Y_mag_residual = torch.cat([discrepancy_Y_mag_residual, new_discrepancy_Y_mag_residual])
        discrepancy_Y = torch.cat([discrepancy_Y, new_discrepancy_Y])
        discrepancy_Y_residual = torch.cat([discrepancy_Y_residual, new_discrepancy_Y_residual])
        discrepancy_Y_total = torch.cat([discrepancy_Y_total, new_discrepancy_Y_total])
        log_discrepancy_Y_total = torch.cat([log_discrepancy_Y_total, new_log_discrepancy_Y_total])
        #jointfit_Y_variance = torch.cat([jointfit_Y_variance,new_jointfit_Y_variance])
        print(f"Iteration {iteration}: best target = {discrepancy_Y_total.min().item():.4f}, new candidate = {candidate}")
        ### check for acceptable candidates
        if log_discrepancy_Y_total_best is None:
            log_discrepancy_Y_total_best = log_discrepancy_Y_total.min().item()
        if log_discrepancy_Y_total_best > new_log_discrepancy_Y_total.item():
            log_discrepancy_Y_total_best = new_log_discrepancy_Y_total.item()
        if iteration == min_discrepancy_iteration:
            condition_firstcheck = (log_discrepancy_Y_total <= log_discrepancy_Y_total_best + logspace_tolerance).squeeze()# if log space chi2 is sufficiently good
            #count = torch.sum(condition_firstcheck).item()
            indices = torch.nonzero(condition_firstcheck)
            #print(indices)
            #now check equivalent chi2 and model error
            #acceptable_candidates = discrepancy_X[condition_firstcheck]
            for idx in indices:
                if (discrepancy_Y_nuc[idx].item()/dof_nuc >= 0.5) & (discrepancy_Y_nuc[idx].item()/dof_nuc <= 2) & (discrepancy_Y_mag[idx].item()/dof_mag >= 0.5) & (discrepancy_Y_mag[idx].item()/dof_mag <= 2):
                    if (discrepancy_X_nuc[idx][-1].item() > 0.01) & (discrepancy_X_nuc[idx][-1].item() < 0.8) (discrepancy_X_mag[idx][-1].item() > 0.01) & (discrepancy_X_mag[idx][-1].item() < 0.8):
                        acceptable_candidates.append(discrepancy_X[idx].tolist())
                        count = count + 1
        if  iteration > min_discrepancy_iteration:
            if new_discrepancy_Y_total <= log_discrepancy_Y_total_best + logspace_tolerance:    
                if (new_discrepancy_Y_nuc.item()/dof_nuc >= 0.5) & (new_discrepancy_Y_nuc.item()/dof_nuc <= 2) & (new_discrepancy_Y_mag.item()/dof_mag >= 0.5) & (new_discrepancy_Y_mag.item()/dof_mag <= 2):
                    if (new_discrepancy_X_nuc[-1].item() > 0.01) & (new_discrepancy_X_nuc[-1].item() < 0.8) (new_discrepancy_X_mag[-1].item() > 0.01) & (new_discrepancy_X_mag[-1].item() < 0.8):
                        count = count + 1
                        acceptable_candidates.append(candidate.tolist())
        if count >= 5:
            #indices = torch.nonzero(condition)
            #print(f"Acceptable fit: chi2_red = {Y.min().item():.3f}")
            break

    best_idx_discrepancy = torch.argmin(discrepancy_Y_total)
    print("Best theta:", discrepancy_X[best_idx_discrepancy])
    print("Best combined chi2:", discrepancy_Y[best_idx_discrepancy])
    print("acceptable candidates:", acceptable_candidates)



    if not acceptable_candidates:
        startpoints_discrepancyfit = [discrepancy_X[best_idx_discrepancy].detach().cpu().numpy()]
    else:
        startpoints_discrepancyfit = np.array(acceptable_candidates)



    ###L-BFGS-B 
    LBFGS_params = params + ['discrepancy_nuc'] + params_mag_1 + ['discrepancy_mag']
    if distribution_type == "Double":
        loss_nuc = make_discrepancy_loss_from_slice(double_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params,fixed_kwargs,LBFGS_params,'discrepancy_nuc')
        loss_mag = make_discrepancy_loss_from_slice(double_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,LBFGS_params,'discrepancy_mag')
        loss_nuc_noresidual = make_discrepancy_loss_from_slice_noresidual(double_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params,fixed_kwargs,LBFGS_params,'discrepancy_nuc')
        loss_mag_noresidual = make_discrepancy_loss_from_slice_noresidual(double_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,LBFGS_params,'discrepancy_mag')
        combined_loss = make_discrepancy_loss(loss_nuc,loss_mag)
        result_LBFGS = LBFGS_optimize(combined_loss,startpoints_discrepancyfit,bounds_for_LBFGS)
    else:
        loss_nuc = make_discrepancy_loss_from_slice(single_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params,fixed_kwargs,LBFGS_params,'discrepancy_nuc')
        loss_mag = make_discrepancy_loss_from_slice(single_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,LBFGS_params,'discrepancy_mag')
        loss_nuc_noresidual = make_discrepancy_loss_from_slice_noresidual(single_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params,fixed_kwargs,LBFGS_params,'discrepancy_nuc')
        loss_mag_noresidual = make_discrepancy_loss_from_slice_noresidual(single_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,LBFGS_params,'discrepancy_mag')
        combined_loss = make_discrepancy_loss(loss_nuc,loss_mag)
        result_LBFGS = LBFGS_optimize(combined_loss,startpoints_discrepancyfit,bounds_for_LBFGS)

    ###Check L-BFGS results###
    acceptable_points = []
    acceptable_equivalent_chi2_nuc = []
    acceptable_equivalent_chi2_mag = []
    acceptable_model_discrepancy_nuc = []
    acceptable_model_discrepancy_mag = []
    for points in result_LBFGS:
        points_nuc = points[:num_parameters+1]
        points_mag = copy.deepcopy(points_nuc)
        for j, src_col in zip(nuc_pos_list, mag_pos_list):
            points_mag[j] = points[num_parameters+1+src_col]
        points_mag[-1] = points[-1]
        if log_Ibg_mag is None:
            points_mag = points_mag[2:]
        equivalent_chi2_nuc = loss_nuc_noresidual(points)
        model_discrepancy_nuc = np.exp(points_nuc[-1])
        equivalent_chi2_mag = loss_mag_noresidual(points)
        model_discrepancy_mag = np.exp(points_mag[-1])
        if equivalent_chi2_nuc/dof_nuc >= 0.5 and equivalent_chi2_nuc/dof_nuc <= 2 and equivalent_chi2_mag/dof_mag >= 0.5 and equivalent_chi2_mag/dof_mag <= 2:
            if abs(model_discrepancy_nuc-sigma_z_nuc) <= sigma_z_nuc and abs(model_discrepancy_mag-sigma_z_mag) <= sigma_z_mag:
                acceptable_points.append(points)
                acceptable_equivalent_chi2_nuc.append(equivalent_chi2_nuc)
                acceptable_equivalent_chi2_mag.append(equivalent_chi2_mag)
                acceptable_model_discrepancy_nuc.append(model_discrepancy_nuc)
                acceptable_model_discrepancy_mag.append(model_discrepancy_mag)

    if acceptable_points:
        print("Found acceptable points in discrepancy fit. Returning point info as output.")
        original_file = st.session_state["data_loader"].name
        original_file_name = original_file.replace(".txt","")
        output_file_name = f"{original_file_name}_Joint_Analysis_secondstage"
        output_file = output_file_name.replace(" ","_") + ".txt"
        filepath = os.path.join(
            OUTPUT_DIR,
            output_file
        )
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"#Original file: {original_file} \n")
            f.write(f"### Analysis Options: ### \n")
            f.write(f"#Distribution Type: {distribution_type} \n")
            f.write(f"#First distribution: {global_model_state["distribution_1"]} \n")
            f.write(f"#First model: {global_model_state["model_1"]} \n")
            if distribution_type == "Double":
                f.write(f"#Second distribution: {global_model_state["distribution_2"]} \n")
                f.write(f"#Second model: {global_model_state["model_2"]} \n")
            f.write(f"#Magnetic Scattering Background: {magnetic_scattering_background} \n")
            f.write(f"### Fit Satisfactory at stage 2: Discrepancy Fit ### \n")
            f.write(f"### Fit Results: ### \n")
            for point in range(len(acceptable_points)):
                f.write(f"### Acceptable Point {point+1} ### \n")
                for idx in range(len(LBFGS_params)):
                    variable_name = LBFGS_params[idx]
                    variable_value = acceptable_points[point][idx]
                    f.write(f"#{variable_name}: {variable_value}\n")
                f.write(f"#Equivalent chi2_red_nuc: {acceptable_equivalent_chi2_nuc[point]/dof_nuc} \n" )
                f.write(f"#Equivalent chi2_red_mag: {acceptable_equivalent_chi2_mag[point]/dof_mag} \n" )
                f.write(f"#True model discrepancy nuc: {acceptable_model_discrepancy_nuc[point]} \n")
                f.write(f"#True model discrepancy mag: {acceptable_model_discrepancy_mag[point]} \n")
        print("Joint Analysis finished!")
        return acceptable_points, acceptable_equivalent_chi2_nuc, acceptable_equivalent_chi2_mag, acceptable_model_discrepancy_nuc, acceptable_model_discrepancy_mag
    
    ### Third Stage: Pareto Sweep ###
    print("Discrepancy fit results not satisfactory. Resorting to Pareto Sweep.")
    ### resolve nuc and mag fitting with fixed params
    fixed_args_nuc_complement = copy.deepcopy(fixed_kwargs)#dynamic
    fixed_args_mag_complement = copy.deepcopy(fixed_kwargs)#dynamic
    nuc_params_temp = np.array(params)
    mag_params_temp = np.array(params_mag)
    jointstart_nuc_filtered = jointstart_nuc[mask_params_nuc]#dynamic
    jointstart_mag_filtered = jointstart_mag[mask_params_mag]#dynamic
    mask_params_nuc_del = np.isin(nuc_params_temp,common_params)
    mask_params_mag_del = np.isin(mag_params_temp,common_params)
    deleted_nuc_values = jointstart_nuc[mask_params_nuc_del]#dynamic
    deleted_mag_values = jointstart_mag[mask_params_mag_del]#dynamic
    #fill dummy values for dicts
    if distribution_type == "Double":
        for entry in possible_common_params_double:
            fixed_args_nuc_complement[entry] = None
            fixed_args_mag_complement[entry] = None
    else:
        for entry in possible_common_params_single:
            fixed_args_nuc_complement[entry] = None
            fixed_args_mag_complement[entry] = None


    for s,v in zip(deleted_mag_params,deleted_mag_values):# for nuc fixed fitting, use mag results, and vice versa
        fixed_args_nuc_complement[s] = v#dynamic

    for s,v in zip(deleted_nuc_params,deleted_nuc_values):
        fixed_args_mag_complement[s] = v#dynamic

    X_filtered_nuc = bounds_nuc_filtered[0]+(bounds_nuc_filtered[1]-bounds_nuc_filtered[0])*torch.rand(n_init_fine,bounds_nuc_filtered.shape[1])
    X_filtered_mag = bounds_mag_filtered[0]+(bounds_mag_filtered[1]-bounds_mag_filtered[0])*torch.rand(n_init_fine,bounds_mag_filtered.shape[1])
    if distribution_type == "Double":
        Y_filtered_nuc = chi2_final_intensity_spheroid_BO_double_fixed(X_filtered_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                    test_Inuc,test_sigmaInuc,fixed_args_nuc_complement['R'],
                                                                    fixed_args_nuc_complement['R_2'],
                                                                    sigma_1=fixed_args_nuc_complement['sigma'],
                                                                    k_1=fixed_args_nuc_complement['k'],
                                                                    sigma_2=fixed_args_nuc_complement['sigma_2'],
                                                                    k_2=fixed_args_nuc_complement['k_2'])
        if log_Ibg_mag is not None:
            Y_filtered_mag = chi2_final_intensity_spheroid_BO_double_fixed(X_filtered_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                    test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                    fixed_args_mag_complement['R_2'],
                                                                    sigma_1=fixed_args_mag_complement['sigma'],
                                                                    k_1=fixed_args_mag_complement['k'],
                                                                    sigma_2=fixed_args_mag_complement['sigma_2'],
                                                                    k_2=fixed_args_mag_complement['k_2'])
        else:
            Y_filtered_mag = chi2_nano_intensity_spheroid_BO_double_fixed(X_filtered_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                    test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                    fixed_args_mag_complement['R_2'],
                                                                    sigma_1=fixed_args_mag_complement['sigma'],
                                                                    k_1=fixed_args_mag_complement['k'],
                                                                    sigma_2=fixed_args_mag_complement['sigma_2'],
                                                                    k_2=fixed_args_mag_complement['k_2'])
    else:
        Y_filtered_nuc = chi2_final_intensity_spheroid_BO_single_fixed(X_filtered_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,
                                                                    test_Inuc,test_sigmaInuc,fixed_args_nuc_complement['R'],
                                                                    sigma=fixed_args_nuc_complement['sigma'],
                                                                    k=fixed_args_nuc_complement['k'])
        if log_Ibg_mag is not None:
            Y_filtered_mag = chi2_final_intensity_spheroid_BO_single_fixed(X_filtered_mag,distribution_func_1,formfactor_1,volume_1,test_Q,
                                                                    test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                    sigma=fixed_args_mag_complement['sigma'],
                                                                    k=fixed_args_mag_complement['k'])
        else:
            Y_filtered_mag = chi2_nano_intensity_spheroid_BO_single_fixed(X_filtered_mag,distribution_func_1,formfactor_1,volume_1,test_Q,
                                                                    test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                    sigma=fixed_args_mag_complement['sigma'],
                                                                    k=fixed_args_mag_complement['k'])
            
    Y_filtered_nuc = Y_filtered_nuc.unsqueeze(-1)
    Y_filtered_mag = Y_filtered_mag.unsqueeze(-1)

    log_Y_filtered_nuc = torch.log(Y_filtered_nuc)
    log_Y_filtered_mag = torch.log(Y_filtered_mag)

    for iteration in range(max_iteration_BO_Pareto):
        model_gp = build_model_nonoise(X_filtered_nuc,log_Y_filtered_nuc,6)
        best_f = log_Y_filtered_nuc.min().item()
        acq = LogExpectedImprovement(model_gp,best_f,maximize=False)#no need to add noise, the model is already imperfect enough
        candidate,_ = optimize_acqf(
            acq_function=acq,
            bounds = bounds_nuc_filtered,
            q = 1,
            num_restarts=10,
            raw_samples=256,
        )
        if distribution_type == "Double":
            new_Y_filtered_nuc = chi2_final_intensity_spheroid_BO_double_fixed(candidate,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                        test_Inuc,test_sigmaInuc,fixed_args_nuc_complement['R'],
                                                                        fixed_args_nuc_complement['R_2'],
                                                                        sigma_1=fixed_args_nuc_complement['sigma'],
                                                                        k_1=fixed_args_nuc_complement['k'],
                                                                        sigma_2=fixed_args_nuc_complement['sigma_2'],
                                                                        k_2=fixed_args_nuc_complement['k_2'])
        else:
            new_Y_filtered_nuc = chi2_final_intensity_spheroid_BO_single_fixed(candidate,distribution_func_1,formfactor_1,volume_1,test_Q,
                                                                        test_Inuc,test_sigmaInuc,fixed_args_nuc_complement['R'],
                                                                        sigma=fixed_args_nuc_complement['sigma'],
                                                                        k=fixed_args_nuc_complement['k'])
        new_Y_filtered_nuc = new_Y_filtered_nuc.unsqueeze(-1)
        new_log_Y_filtered_nuc = torch.log(new_Y_filtered_nuc)
        X_filtered_nuc = torch.cat([X_filtered_nuc,candidate])
        Y_filtered_nuc = torch.cat([Y_filtered_nuc,new_Y_filtered_nuc])
        log_Y_filtered_nuc = torch.cat([log_Y_filtered_nuc,new_log_Y_filtered_nuc])
        print(f"Iteration {iteration}: best Y_filtered_nuc = {Y_filtered_nuc.min().item():.4f}, new candidate = {candidate}")
    best_filtered_nuc_idx = torch.argmin(Y_filtered_nuc)
    BO_filtered_nuc_candidate = [X_filtered_nuc[best_filtered_nuc_idx].detach().cpu().numpy()]
    if distribution_type == "Double":
        filtered_res_nuc = LM_optimize(double_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params=nuc_params_filtered,kwargs=fixed_args_nuc_complement,startpoints=BO_filtered_nuc_candidate)
    else:
        filtered_res_nuc = LM_optimize(single_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params=nuc_params_filtered,kwargs=fixed_args_nuc_complement,startpoints=BO_filtered_nuc_candidate)
    print(f"Complement result nuc: Parameters = {filtered_res_nuc.x}, chi2_red = {filtered_res_nuc.cost*2/dof_nuc:.4f}")

    filtered_res_nuc_np = np.array(filtered_res_nuc.x)
    assembled_theta_magside = assemble_theta(jointfit_params,nuc_params_filtered,filtered_res_nuc_np,mag_params_filtered,jointstart_mag_filtered,fixed_args_nuc_complement)
    print("Lamda = 0 (only magnetic influence) fit result:", assembled_theta_magside)
    nuc_params_lambda0, mag_params_lambda0 = separate_nuc_mag_parameters(assembled_theta_magside,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
    if distribution_type == "Double":
        assembled_theta_magside_chi2_nuc = chi2_final_intensity_spheroid_double(nuc_params_lambda0,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
        if log_Ibg_mag is not None:
            assembled_theta_magside_chi2_mag = chi2_final_intensity_spheroid_double(mag_params_lambda0,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            assembled_theta_magside_chi2_mag = chi2_nano_intensity_spheroid_double(mag_params_lambda0,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        assembled_theta_magside_chi2_nuc = chi2_final_intensity_spheroid_single(nuc_params_lambda0,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
        if log_Ibg_mag is not None:
            assembled_theta_magside_chi2_mag = chi2_final_intensity_spheroid_single(mag_params_lambda0,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        else:
            assembled_theta_magside_chi2_mag = chi2_nano_intensity_spheroid_single(mag_params_lambda0,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    

    ### Now do the same for mag scattering


    for iteration in range(max_iteration_BO_Pareto):
        model_gp = build_model_nonoise(X_filtered_mag,log_Y_filtered_mag,6)
        best_f = log_Y_filtered_mag.min().item()
        acq = LogExpectedImprovement(model_gp,best_f,maximize=False)#no need to add noise, the model is already imperfect enough
        candidate,_ = optimize_acqf(
            acq_function=acq,
            bounds = bounds_mag_filtered,
            q = 1,
            num_restarts=10,
            raw_samples=256,
        )
        if distribution_type == "Double":
            if log_Ibg_mag is not None:
                new_Y_filtered_mag = chi2_final_intensity_spheroid_BO_double_fixed(candidate,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                        test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                        fixed_args_mag_complement['R_2'],
                                                                        sigma_1=fixed_args_mag_complement['sigma'],
                                                                        k_1=fixed_args_mag_complement['k'],
                                                                        sigma_2=fixed_args_mag_complement['sigma_2'],
                                                                        k_2=fixed_args_mag_complement['k_2'])
            else:
                new_Y_filtered_mag = chi2_nano_intensity_spheroid_BO_double_fixed(candidate,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                        test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                        fixed_args_mag_complement['R_2'],
                                                                        sigma_1=fixed_args_mag_complement['sigma'],
                                                                        k_1=fixed_args_mag_complement['k'],
                                                                        sigma_2=fixed_args_mag_complement['sigma_2'],
                                                                        k_2=fixed_args_mag_complement['k_2'])
        else:
            if log_Ibg_mag is not None:
                new_Y_filtered_mag = chi2_final_intensity_spheroid_BO_single_fixed(candidate,distribution_func_1,formfactor_1,volume_1,test_Q,
                                                                        test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                        sigma=fixed_args_mag_complement['sigma'],
                                                                        k=fixed_args_mag_complement['k'])
            else:
                new_Y_filtered_mag = chi2_nano_intensity_spheroid_BO_single_fixed(candidate,distribution_func_1,formfactor_1,volume_1,test_Q,
                                                                        test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                        sigma=fixed_args_mag_complement['sigma'],
                                                                        k=fixed_args_mag_complement['k'])
        new_Y_filtered_mag = new_Y_filtered_mag.unsqueeze(-1)
        new_log_Y_filtered_mag = torch.log(new_Y_filtered_mag)
        X_filtered_mag = torch.cat([X_filtered_mag,candidate])
        Y_filtered_mag = torch.cat([Y_filtered_mag,new_Y_filtered_mag])
        log_Y_filtered_mag = torch.cat([log_Y_filtered_mag,new_log_Y_filtered_mag])
        print(f"Iteration {iteration}: best Y_filtered_mag = {Y_filtered_mag.min().item():.4f}, new candidate = {candidate}")

    best_filtered_mag_idx = torch.argmin(Y_filtered_mag)
    BO_filtered_mag_candidate = [X_filtered_mag[best_filtered_mag_idx].detach().cpu().numpy()]
    if distribution_type == "Double":
        filtered_res_mag = LM_optimize(double_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params=mag_params_filtered,kwargs=fixed_args_mag_complement,startpoints=BO_filtered_mag_candidate)
    else:
        filtered_res_mag = LM_optimize(single_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params=mag_params_filtered,kwargs=fixed_args_mag_complement,startpoints=BO_filtered_mag_candidate)
    print(f"Complement result mag: Parameters = {filtered_res_mag.x}, chi2_red = {filtered_res_mag.cost*2/dof_mag:.4f}")

    filtered_res_mag_np = np.array(filtered_res_mag.x)
    assembled_theta_nucside = assemble_theta(jointfit_params,mag_params_filtered,filtered_res_mag_np,nuc_params_filtered,jointstart_nuc_filtered,fixed_args_mag_complement)
    print("Lamda = 1 (only nuclear influence) fit result:", assembled_theta_nucside)
    nuc_params_lambda1, mag_params_lambda1 = separate_nuc_mag_parameters(assembled_theta_nucside,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
    if distribution_type == "Double":
        assembled_theta_nucside_chi2_nuc = chi2_final_intensity_spheroid_double(nuc_params_lambda1,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
        if log_Ibg_mag is not None:
            assembled_theta_nucside_chi2_mag = chi2_final_intensity_spheroid_double(mag_params_lambda1,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            assembled_theta_nucside_chi2_mag = chi2_nano_intensity_spheroid_double(mag_params_lambda1,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        assembled_theta_nucside_chi2_nuc = chi2_final_intensity_spheroid_single(nuc_params_lambda1,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
        if log_Ibg_mag is not None:
            assembled_theta_nucside_chi2_mag = chi2_final_intensity_spheroid_single(mag_params_lambda1,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        else:
            assembled_theta_nucside_chi2_mag = chi2_nano_intensity_spheroid_single(mag_params_lambda1,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    
    ###### Begin Pareto Front Sweep ######
    # construct starting theta

    #current_theta = construct_joint_parameters(jointstart,jointstart_mag,pos_array_A_2,pos_array_mu_1,pos_array_mu_2,log_Ibg_mag,distribution_type,model_1,model_2)
    #print(current_theta)
    current_theta = [assembled_theta_magside]
    #construct lambda checklist
    list_lambda = np.concatenate((np.linspace(0.1,0.3,2,endpoint=False),np.linspace(0.3,0.7,8,endpoint=False),np.linspace(0.7,1.0,3,endpoint=False)))
    list_weights = []
    list_pareto_candidates = []
    list_pareto_chi2_red_nuc = []
    list_pareto_chi2_red_mag = []
    list_weights.append(0)
    list_pareto_candidates.append(assembled_theta_magside)
    list_pareto_chi2_red_nuc.append(assembled_theta_magside_chi2_nuc)
    list_pareto_chi2_red_mag.append(assembled_theta_magside_chi2_mag)

    list_weights.append(1)
    list_pareto_candidates.append(assembled_theta_nucside)
    list_pareto_chi2_red_nuc.append(assembled_theta_nucside_chi2_nuc)
    list_pareto_chi2_red_mag.append(assembled_theta_nucside_chi2_mag)

    ### Start Pareto Sweep ###
    for weight in list_lambda:
        ### first do LM to approach solution, start sweep from lambda = 0 (magnetic scattering)
        if distribution_type == "Double":
            residual_nuc_sweep = make_residuals_from_slice(double_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params,fixed_kwargs,jointfit_params)
            residual_mag_sweep = make_residuals_from_slice(double_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
        else:
            residual_nuc_sweep = make_residuals_from_slice(single_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params,fixed_kwargs,jointfit_params)
            residual_mag_sweep = make_residuals_from_slice(single_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
        joint_residual_sweep = make_weighted_joint_residuals(residual_nuc_sweep,residual_mag_sweep,weight,dof_nuc,dof_mag)
        joint_residual_sweep_nostop = make_weighted_joint_residuals(residual_nuc_sweep,residual_mag_sweep,weight,dof_nuc,dof_mag,stop=False)
        result_joint_weighted_firstfit = LM_joint_optimize(joint_residual_sweep,joint_residual_sweep_nostop,startpoints=current_theta)
        print("weight:",weight, "First approximation:", result_joint_weighted_firstfit.x)
        ### do a fine BO using this starting point
        BO_startpoint_nuc, BO_startpoint_mag = separate_nuc_mag_parameters(np.array(result_joint_weighted_firstfit.x),num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
        bounds_BO_sweep = construct_joint_bounds(BO_startpoint_nuc,BO_startpoint_mag,joint_searchwidth_fine,pos_array_A_2,pos_array_mu_1,pos_array_mu_2,log_Ibg_mag,distribution_type,model_1,model_2)
        X_sweep = bounds_BO_sweep[0]+(bounds_BO_sweep[1]-bounds_BO_sweep[0])*torch.rand(n_init_fine,bounds_BO_sweep.shape[1])
        X_sweep_nuc, X_sweep_mag = separate_nuc_mag_parameters_tensor(X_sweep,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
        if distribution_type == "Double":
            Y_sweep_nuc = chi2_final_intensity_spheroid_BO_double(X_sweep_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None:
                Y_sweep_mag = chi2_final_intensity_spheroid_BO_double(X_sweep_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                Y_sweep_mag = chi2_nano_intensity_spheroid_BO_double(X_sweep_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            Y_sweep_nuc = chi2_final_intensity_spheroid_BO_single(X_sweep_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None: 
                Y_sweep_mag = chi2_final_intensity_spheroid_BO_single(X_sweep_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            else:
                Y_sweep_mag = chi2_nano_intensity_spheroid_BO_single(X_sweep_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        
        Y_sweep_nuc = Y_sweep_nuc.unsqueeze(-1)
        Y_sweep_mag = Y_sweep_mag.unsqueeze(-1)
        Y_sweep = Y_sweep_nuc*weight + Y_sweep_mag*(1-weight)
        log_Y_sweep = torch.log(Y_sweep)
        for iteration in range(max_iteration_BO_Pareto):
            model_gp = build_model_nonoise(X_sweep,log_Y_sweep,beta_jointfit)
            best_f = log_Y_sweep.min().item()
            acq = LogExpectedImprovement(model_gp,best_f,maximize=False)
            candidate,_ = optimize_acqf(
            acq_function=acq,
            bounds = bounds_BO_sweep,
            q = 1,
            num_restarts=10,
            raw_samples=256,
        )
            new_X_sweep_nuc, new_X_sweep_mag = separate_nuc_mag_parameters_tensor(candidate,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
            if distribution_type == "Double":
                new_Y_sweep_nuc = chi2_final_intensity_spheroid_BO_double(new_X_sweep_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
                if log_Ibg_mag is not None:
                    new_Y_sweep_mag = chi2_final_intensity_spheroid_BO_double(new_X_sweep_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
                else:
                    new_Y_sweep_mag = chi2_nano_intensity_spheroid_BO_double(new_X_sweep_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                new_Y_sweep_nuc = chi2_final_intensity_spheroid_BO_single(new_X_sweep_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
                if log_Ibg_mag is not None: 
                    new_Y_sweep_mag = chi2_final_intensity_spheroid_BO_single(new_X_sweep_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
                else:
                    new_Y_sweep_mag = chi2_nano_intensity_spheroid_BO_single(new_X_sweep_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            new_Y_sweep_nuc = new_Y_sweep_nuc.unsqueeze(-1)
            new_Y_sweep_mag = new_Y_sweep_mag.unsqueeze(-1)
            new_Y_sweep = new_Y_sweep_nuc*weight + new_Y_sweep_mag*(1-weight)
            new_log_Y_sweep = torch.log(new_Y_sweep)
            X_sweep = torch.cat([X_sweep,candidate])
            Y_sweep_nuc = torch.cat([Y_sweep_nuc,new_Y_sweep_nuc])
            Y_sweep_mag = torch.cat([Y_sweep_mag,new_Y_sweep_mag])
            Y_sweep = torch.cat([Y_sweep,new_Y_sweep])
            log_Y_sweep = torch.cat([log_Y_sweep,new_log_Y_sweep])
            print(f"Iteration {iteration}, Weight {weight}: best sweep target = {Y_sweep.min().item():.4f}, new candidate = {candidate}")

        best_idx_current_BO = torch.argmin(Y_sweep)
        print(f"Weight {weight}: best theta={X_sweep[best_idx_current_BO]}, best weighted chi2={Y_sweep[best_idx_current_BO]}")
        current_BO_result = [X_sweep[best_idx_current_BO].detach().cpu().numpy()]
        #rerun local optimizer
        result_joint_weighted_finalfit = LM_joint_optimize(joint_residual_sweep,joint_residual_sweep_nostop,startpoints=current_BO_result)
        #output,update for next lambda
        print("Current weight:", weight, "Best theta:", result_joint_weighted_finalfit.x)
        list_weights.append(weight)
        list_pareto_candidates.append(result_joint_weighted_finalfit.x)
        result_joint_weighted_finalfit_nuc,result_joint_weighted_finalfit_mag = separate_nuc_mag_parameters(np.array(result_joint_weighted_finalfit.x),num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
        if distribution_type == "Double":
            chi2_joint_weighted_finalfit_nuc = chi2_final_intensity_spheroid_double(result_joint_weighted_finalfit_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None:
                chi2_joint_weighted_finalfit_mag = chi2_final_intensity_spheroid_double(result_joint_weighted_finalfit_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                chi2_joint_weighted_finalfit_mag = chi2_nano_intensity_spheroid_double(result_joint_weighted_finalfit_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            chi2_joint_weighted_finalfit_nuc = chi2_final_intensity_spheroid_single(result_joint_weighted_finalfit_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None:
                chi2_joint_weighted_finalfit_mag = chi2_final_intensity_spheroid_single(result_joint_weighted_finalfit_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            else:
                chi2_joint_weighted_finalfit_mag = chi2_nano_intensity_spheroid_single(result_joint_weighted_finalfit_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        list_pareto_chi2_red_nuc.append(chi2_joint_weighted_finalfit_nuc)
        list_pareto_chi2_red_mag.append(chi2_joint_weighted_finalfit_mag)
        current_theta = [np.array(result_joint_weighted_finalfit.x)]
        
    #Do a reverse sweep as well
    current_theta = [assembled_theta_nucside]
    for weight in list_lambda:
        ### first do LM to approach solution, start sweep from lambda = 1 (nuclear scattering)
        if distribution_type == "Double":
            residual_nuc_sweep = make_residuals_from_slice(double_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params,fixed_kwargs,jointfit_params)
            residual_mag_sweep = make_residuals_from_slice(double_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
        else:
            residual_nuc_sweep = make_residuals_from_slice(single_intensity_spheroid,test_Q,'Q',test_Inuc,test_sigmaInuc,params,fixed_kwargs,jointfit_params)
            residual_mag_sweep = make_residuals_from_slice(single_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
        joint_residual_sweep = make_weighted_joint_residuals(residual_mag_sweep,residual_nuc_sweep,weight,dof_mag,dof_nuc)#inversing order of residual input = reverse sweep
        joint_residual_sweep_nostop = make_weighted_joint_residuals(residual_mag_sweep,residual_nuc_sweep,weight,dof_mag,dof_nuc,stop=False)
        result_joint_weighted_firstfit = LM_joint_optimize(joint_residual_sweep,joint_residual_sweep_nostop,startpoints=current_theta)
        print("weight:",weight, "First approximation:", result_joint_weighted_firstfit.x)
        ### do a fine BO using this starting point
        BO_startpoint_nuc, BO_startpoint_mag = separate_nuc_mag_parameters(np.array(result_joint_weighted_firstfit.x),num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
        bounds_BO_sweep = construct_joint_bounds(BO_startpoint_nuc,BO_startpoint_mag,joint_searchwidth_fine,pos_array_A_2,pos_array_mu_1,pos_array_mu_2,log_Ibg_mag,distribution_type,model_1,model_2)
        X_sweep = bounds_BO_sweep[0]+(bounds_BO_sweep[1]-bounds_BO_sweep[0])*torch.rand(n_init_fine,bounds_BO_sweep.shape[1])
        X_sweep_nuc, X_sweep_mag = separate_nuc_mag_parameters_tensor(X_sweep,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
        if distribution_type == "Double":
            Y_sweep_nuc = chi2_final_intensity_spheroid_BO_double(X_sweep_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None:
                Y_sweep_mag = chi2_final_intensity_spheroid_BO_double(X_sweep_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                Y_sweep_mag = chi2_nano_intensity_spheroid_BO_double(X_sweep_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            Y_sweep_nuc = chi2_final_intensity_spheroid_BO_single(X_sweep_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None: 
                Y_sweep_mag = chi2_final_intensity_spheroid_BO_single(X_sweep_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            else:
                Y_sweep_mag = chi2_nano_intensity_spheroid_BO_single(X_sweep_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        
        Y_sweep_nuc = Y_sweep_nuc.unsqueeze(-1)
        Y_sweep_mag = Y_sweep_mag.unsqueeze(-1)
        Y_sweep = Y_sweep_nuc*(1-weight) + Y_sweep_mag*weight
        log_Y_sweep = torch.log(Y_sweep)
        for iteration in range(max_iteration_BO_Pareto):
            model_gp = build_model_nonoise(X_sweep,log_Y_sweep,beta_jointfit)
            best_f = log_Y_sweep.min().item()
            acq = LogExpectedImprovement(model_gp,best_f,maximize=False)
            candidate,_ = optimize_acqf(
            acq_function=acq,
            bounds = bounds_BO_sweep,
            q = 1,
            num_restarts=10,
            raw_samples=256,
        )
            new_X_sweep_nuc, new_X_sweep_mag = separate_nuc_mag_parameters_tensor(candidate,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
            if distribution_type == "Double":
                new_Y_sweep_nuc = chi2_final_intensity_spheroid_BO_double(new_X_sweep_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
                if log_Ibg_mag is not None:
                    new_Y_sweep_mag = chi2_final_intensity_spheroid_BO_double(new_X_sweep_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
                else:
                    new_Y_sweep_mag = chi2_nano_intensity_spheroid_BO_double(new_X_sweep_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                new_Y_sweep_nuc = chi2_final_intensity_spheroid_BO_single(new_X_sweep_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
                if log_Ibg_mag is not None: 
                    new_Y_sweep_mag = chi2_final_intensity_spheroid_BO_single(new_X_sweep_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
                else:
                    new_Y_sweep_mag = chi2_nano_intensity_spheroid_BO_single(new_X_sweep_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            new_Y_sweep_nuc = new_Y_sweep_nuc.unsqueeze(-1)
            new_Y_sweep_mag = new_Y_sweep_mag.unsqueeze(-1)
            new_Y_sweep = new_Y_sweep_nuc*(1-weight) + new_Y_sweep_mag*weight
            new_log_Y_sweep = torch.log(new_Y_sweep)
            X_sweep = torch.cat([X_sweep,candidate])
            Y_sweep_nuc = torch.cat([Y_sweep_nuc,new_Y_sweep_nuc])
            Y_sweep_mag = torch.cat([Y_sweep_mag,new_Y_sweep_mag])
            Y_sweep = torch.cat([Y_sweep,new_Y_sweep])
            log_Y_sweep = torch.cat([log_Y_sweep,new_log_Y_sweep])
            print(f"Iteration {iteration}, Weight {weight}: best sweep target = {Y_sweep.min().item():.4f}, new candidate = {candidate}")

        best_idx_current_BO = torch.argmin(Y_sweep)
        print(f"Weight {weight}: best theta={X_sweep[best_idx_current_BO]}, best weighted chi2={Y_sweep[best_idx_current_BO]}")
        current_BO_result = [X_sweep[best_idx_current_BO].detach().cpu().numpy()]
        #rerun local optimizer
        result_joint_weighted_finalfit = LM_joint_optimize(joint_residual_sweep,joint_residual_sweep_nostop,startpoints=current_BO_result)
        #output,update for next lambda
        print("Current weight:", weight, "Best theta:", result_joint_weighted_finalfit.x)
        list_weights.append(1-weight)
        list_pareto_candidates.append(result_joint_weighted_finalfit.x)
        result_joint_weighted_finalfit_nuc,result_joint_weighted_finalfit_mag = separate_nuc_mag_parameters(np.array(result_joint_weighted_finalfit.x),num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
        if distribution_type == "Double":
            chi2_joint_weighted_finalfit_nuc = chi2_final_intensity_spheroid_double(result_joint_weighted_finalfit_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None:
                chi2_joint_weighted_finalfit_mag = chi2_final_intensity_spheroid_double(result_joint_weighted_finalfit_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                chi2_joint_weighted_finalfit_mag = chi2_nano_intensity_spheroid_double(result_joint_weighted_finalfit_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            chi2_joint_weighted_finalfit_nuc = chi2_final_intensity_spheroid_single(result_joint_weighted_finalfit_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
            if log_Ibg_mag is not None:
                chi2_joint_weighted_finalfit_mag = chi2_final_intensity_spheroid_single(result_joint_weighted_finalfit_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            else:
                chi2_joint_weighted_finalfit_mag = chi2_nano_intensity_spheroid_single(result_joint_weighted_finalfit_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        list_pareto_chi2_red_nuc.append(chi2_joint_weighted_finalfit_nuc)
        list_pareto_chi2_red_mag.append(chi2_joint_weighted_finalfit_mag)
        current_theta = [np.array(result_joint_weighted_finalfit.x)]
    ###  Analyze sweep result  ###
    #first discard less ideal points
    keep = [True]*len(list_pareto_candidates)
    for i in range(len(list_pareto_candidates)):
        for j in range(len(list_pareto_candidates)):
            if i == j:
                continue
            if (list_pareto_chi2_red_nuc[j] <= list_pareto_chi2_red_nuc[i] and list_pareto_chi2_red_mag[j] <= list_pareto_chi2_red_mag[i]) and (list_pareto_chi2_red_nuc[j] < list_pareto_chi2_red_nuc[i] or list_pareto_chi2_red_mag[j] < list_pareto_chi2_red_mag[i]):
                keep[i] = False
                break
    list_pareto_front = [list_pareto_candidates[i] for i in range(len(list_pareto_candidates)) if keep[i]]
    list_pareto_front_chi2_red_nuc = [list_pareto_chi2_red_nuc[i] for i in range(len(list_pareto_candidates)) if keep[i]]
    list_pareto_front_chi2_red_mag = [list_pareto_chi2_red_mag[i] for i in range(len(list_pareto_candidates)) if keep[i]]
    print("pareto set:", list_pareto_front)
    print("chi2 red nuc values:", list_pareto_front_chi2_red_nuc)
    print("chi2 red mag values:", list_pareto_front_chi2_red_mag)
    #then find pareto best solution

    points = sorted(zip(list_pareto_front_chi2_red_nuc,list_pareto_front,list_pareto_front_chi2_red_mag,list_weights),key = lambda x: x[0])
    list_pareto_front_chi2_red_nuc_sorted = [p[0] for p in points]
    list_pareto_front_sorted = [p[1] for p in points]
    list_pareto_front_chi2_red_mag_sorted = [p[2] for p in points]
    list_weights_sorted = [p[3] for p in points]
    if len(list_pareto_front_chi2_red_nuc_sorted) <= 3:
        print("Not enough pareto solutions. Returning all parameter sets.") 
    else:
        x1, y1 = list_pareto_front_chi2_red_nuc_sorted[0],  list_pareto_front_chi2_red_mag_sorted[0]
        x2, y2 = list_pareto_front_chi2_red_nuc_sorted[-1], list_pareto_front_chi2_red_mag_sorted[-1]
        denom = math.hypot(x2-x1,y2-y1)
        if denom == 0:
            print("Points overlap. Check data")
        else:
            max_dist = -1.0
            best_idx = -1
            for i in range(len(list_pareto_front_chi2_red_nuc_sorted)):
                x0, y0 = list_pareto_front_chi2_red_nuc_sorted[i], list_pareto_front_chi2_red_mag_sorted[i]
                dist = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / denom
                if dist > max_dist:
                    max_dist = dist
                    best_idx = i

            print("Best Pareto Solution:", list_pareto_front_sorted[best_idx])
            print("Best chi2_red_nuc:", list_pareto_front_chi2_red_nuc_sorted[best_idx])
            print("Best chi2_red_mag:", list_pareto_front_chi2_red_mag_sorted[best_idx])

    ### Output all pareto solutions ###
    print("Joint Analysis finished. Returning all pareto solutions.")
    original_file = st.session_state["data_loader"].name
    original_file_name = original_file.replace(".txt","")
    output_file_name = f"{original_file_name}_Joint_Analysis_thirdstage"
    output_file = output_file_name.replace(" ","_") + ".txt"
    filepath = os.path.join(
        OUTPUT_DIR,
        output_file
    )
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"#Original file: {original_file} \n")
        f.write(f"### Analysis Options: ### \n")
        f.write(f"#Distribution Type: {distribution_type} \n")
        f.write(f"#First distribution: {global_model_state["distribution_1"]} \n")
        f.write(f"#First model: {global_model_state["model_1"]} \n")
        if distribution_type == "Double":
            f.write(f"#Second distribution: {global_model_state["distribution_2"]} \n")
            f.write(f"#Second model: {global_model_state["model_2"]} \n")
        f.write(f"#Magnetic Scattering Background: {magnetic_scattering_background} \n")
        f.write(f"### Fit Satisfactory at stage 3: Pareto Sweep ### \n")
        f.write(f"### Fit Results: ### \n")
        for parameter in jointfit_params:
            f.write(f"{parameter}")
        f.write(f"weight_nuc chi2_red_nuc chi2_red_mag\n")
        for i in range(len(list_pareto_front_sorted)):
            params = list_pareto_front_sorted[i]
            weight = list_weights_sorted[i]
            chi2_red_nuc = list_pareto_front_chi2_red_nuc_sorted[i]
            chi2_red_mag = list_pareto_front_chi2_red_mag_sorted[i]
            params_str = " ".join([f"{p:.6e}" for p in params])
            f.write(f"{params_str} {weight:.4f} {chi2_red_nuc:.4f} {chi2_red_mag:.4f}")
            if i == best_idx:
                f.write(f"  #***Best Pareto Solution\n")
            else:
                f.write(f"\n")
        if len(list_pareto_front_chi2_red_nuc_sorted) > 3:
            f.write(f"### Knee point solution model intensity: ### \n")
            lookup = dict(zip(jointfit_params, list_pareto_front_sorted[best_idx]))
            log_Ibg = lookup.get("background_log")
            log_C = lookup.get("C_log")
            A = lookup.get("combinedfactor_log")
            Rm_1 = lookup.get("R")
            sigma_1 = lookup.get("sigma")
            k_1 = lookup.get("k")
            mu_1 = lookup.get("mu")
            A_2 = lookup.get("combinedfactor_2_log")
            Rm_2 = lookup.get("R_2")
            sigma_2 = lookup.get("sigma_2")
            k_2 = lookup.get("k_2")
            mu_2 = lookup.get("mu_2")
            log_Ibg_mag = lookup.get("background_log_mag")
            log_C_mag = lookup.get("C_log_mag")
            A_mag = lookup.get("combinedfactor_log_mag")
            A_2_mag = lookup.get("combinedfactor_2_log_mag")
            mu_1_mag = lookup.get("mu_mag")
            mu_2_mag = lookup.get("mu_2_mag")
            combinedfactor_1 = np.exp(-A)
            combinedfactor_1_mag = np.exp(-A_mag)
            if log_Ibg is not None:
                Ibg = np.exp(log_Ibg)
                C = np.exp(log_C)
            if sigma_1 is not None:
                sigma_rm_1 = sigma_1*Rm_1
            if sigma_2 is not None:
                sigma_rm_2 = sigma_2*Rm_2
            if A_2 is not None:
                combinedfactor_2 = np.exp(-A_2)
            if log_Ibg_mag is not None:
                Ibg_mag = np.exp(log_Ibg_mag)
                C_mag = np.exp(log_C_mag)
            if A_2_mag is not None:
                combinedfactor_2_mag = np.exp(-A_2_mag)
            
            if distribution_type == "Double":
                model_Inuc = [(final_intensity_spheroid(
                            I_porod,combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                            Ibg,q,Rm_1,C,sigma=sigma_rm_1,k=k_1,mu=mu_1
                            )+nano_intensity_spheroid(combinedfactor_2,distribution_func_2,formfactor_2,volume_2,
                                                    q,Rm_2,sigma=sigma_rm_2,k=k_2,mu=mu_2)) for q in model_Q]
                if log_Ibg_mag is not None:
                    model_Imag = [(final_intensity_spheroid(
                            I_porod,combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                            Ibg_mag,q,Rm_1,C_mag,sigma=sigma_rm_1,k=k_1,mu=mu_1_mag
                            )+nano_intensity_spheroid(combinedfactor_2_mag,distribution_func_2,formfactor_2,volume_2,
                                                    q,Rm_2,sigma=sigma_rm_2,k=k_2,mu=mu_2_mag)) for q in model_Q]
                else:
                    model_Imag = [(nano_intensity_spheroid(
                            combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                            q,Rm_1,sigma=sigma_rm_1,k=k_1,mu=mu_1_mag
                            )+nano_intensity_spheroid(combinedfactor_2_mag,distribution_func_2,formfactor_2,volume_2,
                                                    q,Rm_2,sigma=sigma_rm_2,k=k_2,mu=mu_2_mag)) for q in model_Q]
            elif distribution_type == "Single":
                model_Inuc = [final_intensity_spheroid(
                            I_porod,combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                            Ibg,q,Rm_1,C,sigma=sigma_rm_1,k=k_1,mu=mu_1
                            ) for q in model_Q]
                if log_Ibg_mag is not None:
                    model_Imag = [final_intensity_spheroid(
                            I_porod,combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                            Ibg_mag,q,Rm_1,C_mag,sigma=sigma_rm_1,k=k_1,mu=mu_1_mag
                            ) for q in model_Q]
                else:                    
                    model_Imag = [nano_intensity_spheroid(
                            combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                            q,Rm_1,sigma=sigma_rm_1,k=k_1,mu=mu_1_mag
                            ) for q in model_Q]
            f.write(f"Q(nm-1) Inuc(Q)(cm-1) Imag(Q)(cm-1)\n")
            for q, Inuc, Imag in zip(model_Q,model_Inuc,model_Imag):
                f.write(f"{q} {Inuc} {Imag}\n")
        else:
            f.write(f"### Not enough pareto solutions to determine knee point. ### \n")
    
    print("Joint Analysis completed.")
    return list_pareto_front_sorted, list_pareto_front_chi2_red_nuc_sorted, list_pareto_front_chi2_red_mag_sorted, list_weights_sorted