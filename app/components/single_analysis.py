import streamlit as st
import plotly.graph_objects as go
import numpy as np
import torch
from botorch.acquisition.analytic import LogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
import os

from problems.approximations import I_porod
from core.models.BO_model import chi2_final_intensity_spheroid_BO_single,chi2_final_intensity_spheroid_BO_double,chi2_nano_intensity_spheroid_BO_double,chi2_nano_intensity_spheroid_BO_single
from core.optimizer.Levenberg_Marquardt import LM_optimize, LM_optimize_nostop
from core.acquisition.acquisition_functions import build_model
from core.models.assembled_problem import final_intensity_spheroid,nano_intensity_spheroid
from app.utilities.cache_generator import get_model_config
from core.utils.helper_functions import log_chi2_red_variance
from core.models.assembled_problem import final_intensity_spheroid,nano_intensity_spheroid,double_intensity_spheroid,single_intensity_spheroid,double_intensity_spheroid_mag,single_intensity_spheroid_mag

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
def singular_fit(treated_input,global_model_state,analysis_configs,active_dataset,variable_values,BO_config,output_graph_data=True):
    ### gather st info ###
    analysis_mode =  global_model_state["analysis_mode"]
    distribution_type = global_model_state["distribution_type"]
    magnetic_scattering_background = global_model_state["magnetic_scattering_background"]
    model_state = get_model_config(global_model_state)
    ### input decode ###
    if distribution_type == "Double":
        distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2 = model_state
        Ibg,_,_,_,_,_,_,_,_,_,_,_,_ = variable_values
    elif distribution_type == "Single":
        distribution_func_1,formfactor_1,volume_1 = model_state
        Ibg,_,_,_,_,_,_,_ = variable_values
        
    if active_dataset == "Magnetic Signal" and magnetic_scattering_background == "Off":
        Ibg = None
        C = None
    if analysis_mode == "Distribution Fitting":
        test_Q, test_Inuc, test_sigmaInuc, test_sigmaQ = treated_input
    elif analysis_mode == "Nuclear-Magnetic Joint Analysis":
        test_Q, test_Inuc, test_sigmaInuc, test_sigmaQ, test_Imag, test_sigmaImag = treated_input
    if active_dataset == "Nuclear Signal":
        test_I = test_Inuc
        test_sigmaI = test_sigmaInuc
    elif active_dataset == "Magnetic Signal":
        test_I = test_Imag
        test_sigmaI = test_sigmaImag
    
    n_init = analysis_configs["n_init"]
    max_iteration_BO = analysis_configs["max_iteration_BO"]

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
    ### case by case determination ###
    if active_dataset == "Nuclear Signal":
        dof = test_I.shape[0]-bounds.shape[1]
        BO_bounds = bounds
        fit_params = params
    elif active_dataset == "Magnetic Signal":
        if Ibg is None:
            dof = test_I.shape[0]-bounds.shape[1] + 2   
        BO_bounds = bounds_mag
        fit_params = params_firstfit_mag
    ### BO preparation ###
    beta = 3.0/(1/BO_bounds.shape[1]**0.5)
    X = BO_bounds[0] + (BO_bounds[1] - BO_bounds[0]) * torch.rand(n_init, BO_bounds.shape[1])
    if distribution_type == "Double":
        if Ibg is not None:
            Y = chi2_final_intensity_spheroid_BO_double(X,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
        else:
            Y = chi2_nano_intensity_spheroid_BO_double(X,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
    else:
        if Ibg is not None:
            Y = chi2_final_intensity_spheroid_BO_single(X,distribution_func_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
        else:
            Y = chi2_nano_intensity_spheroid_BO_single(X,distribution_func_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
    Y = Y.unsqueeze(-1)        
    log_Y = torch.log(Y)
    ### Start BO loop ###
    for iteration in range(max_iteration_BO):
        s2_Y = log_Y.var()
        Y_variance = log_chi2_red_variance(Y,dof,s2=s2_Y)
        model_gp = build_model(X,log_Y,Y_variance,beta)
        acq = LogNoisyExpectedImprovement(model_gp,X,maximize=False)
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=BO_bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        if distribution_type == "Double":
            if Ibg is not None:
                new_Y = chi2_final_intensity_spheroid_BO_double(candidate,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
            else:
                new_Y = chi2_nano_intensity_spheroid_BO_double(candidate,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
        else:
            if Ibg is not None:
                new_Y = chi2_final_intensity_spheroid_BO_single(candidate,distribution_func_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
            else:
                new_Y = chi2_nano_intensity_spheroid_BO_single(candidate,distribution_func_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
        new_Y = new_Y.unsqueeze(-1)
        new_log_Y = torch.log(new_Y)
        # Update dataset
        X = torch.cat([X, candidate])
        Y = torch.cat([Y, new_Y])
        log_Y = torch.cat([log_Y,new_log_Y])        
        print(f"Iteration {iteration}: best Y = {Y.min().item():.4f}, new candidate = {candidate}")
        condition = (Y >= 0.9) & (Y <= 2.0)
        count = torch.sum(condition).item()
        if count >= 3:
            print(f"Acceptable fit: chi2_red = {Y.min().item():.3f}")
            break

    best_idx = torch.argmin(Y)
    print("Best theta:", X[best_idx])
    print("Best chi2:", Y[best_idx])
    condition = condition.squeeze(1)
    startpoints = X[condition]

    if torch.any(condition).item(): 
        condition = condition.squeeze(1)
        startpoints = X[condition]
        if count != 1:
            BO_candidates = startpoints.detach().cpu().numpy()
        else:
            BO_candidates = [X[best_idx].detach().cpu().numpy()]
    else:
        BO_candidates = [X[best_idx].detach().cpu().numpy()]
    
    if distribution_type == "Double":
        #result = LM_optimize(double_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params=fit_params,kwargs=fixed_kwargs,startpoints=BO_candidates)
        result = LM_optimize_nostop(double_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params=fit_params,kwargs=fixed_kwargs,startpoints=BO_candidates)
    else:
        #result = LM_optimize(single_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params=fit_params,kwargs=fixed_kwargs,startpoints=BO_candidates)
        result = LM_optimize_nostop(single_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params=fit_params,kwargs=fixed_kwargs,startpoints=BO_candidates)

    print(f"Best result: Parameters = {result.x}, chi2_red = {result.cost*2/dof:.4f}")
    jointstart = result.x
    J = result.jac #jacobian
    H_approx = J.T @ J #hessian approximation
    cov = np.linalg.inv(H_approx) #covariance matrix
    uncertainties = np.sqrt(np.diag(cov)) #fit uncertainties
    chi2_red = result.cost*2/dof
    ### write data into output file ###
    if output_graph_data:
        lookup = dict(zip(fit_params,jointstart))
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
        combinedfactor_1 = np.exp(-A)
        if log_Ibg is not None:
            Ibg = np.exp(log_Ibg)
            C = np.exp(log_C)
        if sigma_1 is not None:
            sigma_rm_1 = sigma_1*Rm_1
        if sigma_2 is not None:
            sigma_rm_2 = sigma_2*Rm_2
        if A_2 is not None:
            combinedfactor_2 = np.exp(-A_2)
        model_Q = np.linspace(max(np.min(test_Q)-0.1,0.01),np.max(test_Q)+0.1,100)
        if distribution_type == "Double":
            if Ibg is not None:
                model_I = [(final_intensity_spheroid(
                        I_porod,combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                        Ibg,q,Rm_1,C,sigma=sigma_rm_1,k=k_1,mu=mu_1
                        )+nano_intensity_spheroid(combinedfactor_2,distribution_func_2,formfactor_2,volume_2,
                                                q,Rm_2,sigma=sigma_rm_2,k=k_2,mu=mu_2)) for q in model_Q]
            else:
                model_I = [(nano_intensity_spheroid(
                        combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                        q,Rm_1,sigma=sigma_rm_1,k=k_1,mu=mu_1
                        )+nano_intensity_spheroid(combinedfactor_2,distribution_func_2,formfactor_2,volume_2,
                                                q,Rm_2,sigma=sigma_rm_2,k=k_2,mu=mu_2)) for q in model_Q]
        elif distribution_type == "Single":
            if Ibg is not None:
                model_I = [final_intensity_spheroid(
                        I_porod,combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                        Ibg,q,Rm_1,C,sigma=sigma_rm_1,k=k_1,mu=mu_1
                        ) for q in model_Q]
            else:                    
                model_I = [nano_intensity_spheroid(
                        combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                        q,Rm_1,sigma=sigma_rm_1,k=k_1,mu=mu_1
                        ) for q in model_Q]
        original_file = st.session_state["data_loader"].name
        original_file_name = original_file.replace(".txt","")
        output_file_name = f"{original_file_name}_{active_dataset}_Single_Analysis"
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
            f.write(f"### Fit Results: ### \n")
            for idx in range(len(fit_params)):
                variable_name = fit_params[idx]
                variable_value = jointstart[idx]
                variable_uncertainty = uncertainties[idx]
                f.write(f"#{variable_name}: {variable_value}, uncertainty: {variable_uncertainty} \n")
            f.write(f"Chi2_red: {chi2_red} \n" )
            f.write(f"### Model Intensity ### \n")
            f.write(f"Q(nm-1) I(Q)(cm-1) \n")
            for q, Iq in zip(model_Q,model_I):
                f.write(f"{q} {Iq}\n")
    print("Single Analysis Finished!")
    return jointstart, chi2_red


