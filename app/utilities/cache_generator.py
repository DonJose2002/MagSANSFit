import streamlit as st
import numpy as np
import torch
import copy

from problems.coreshell import formfactor_coreshell
from problems.distributions import distribution_normal,distribution_lognormal
from problems.ellipsoid import volume_ellipsoid,formfactor_ellipsoid
from problems.sphere import volume_sphere,formfactor_sphere
from problems.approximations import I_porod
from core.models.assembled_problem import final_intensity_spheroid,nano_intensity_spheroid
from core.models.loss_functions import loss_chi2
from core.models.single_loss import chi2_final_intensity_spheroid_double,chi2_final_intensity_spheroid_single,chi2_nano_intensity_spheroid_double,chi2_nano_intensity_spheroid_single
"""
get_model_config: returns the non-value input that analysis should be using based on model_state.
uses cache_resource decorator to conserve function output (callable objects have problems with cache_data)
"""
@st.cache_resource
def get_model_config(global_model_state):
    distribution_type = global_model_state["distribution_type"]
    distribution_1 = global_model_state["distribution_1"]
    distribution_2 = global_model_state["distribution_2"]
    model_1 = global_model_state["model_1"]
    model_2 = global_model_state["model_2"]
    distribution_func_1 = "mono"
    formfactor_1 = formfactor_sphere
    volume_1 = volume_sphere
    if distribution_1 != "mono":
        if distribution_1 == "log_normal":
            distribution_func_1 = distribution_lognormal
        elif distribution_1 =="normal":
            distribution_func_1 = distribution_normal

    if model_1 == "ellipsoid":
        formfactor_1 = formfactor_ellipsoid
        volume_1 = volume_ellipsoid
    elif model_1 == "core-shell":
        formfactor_1 = formfactor_coreshell
        
    if distribution_type == "Double": 
        distribution_func_2 = "mono"
        formfactor_2 = formfactor_sphere
        volume_2 = volume_sphere
        if distribution_2 != "mono":
            if distribution_2 == "log_normal":
                distribution_func_2 = distribution_lognormal
            elif distribution_2 =="normal":
                distribution_func_2 = distribution_normal

        if model_2 == "ellipsoid":
            formfactor_2 = formfactor_ellipsoid
            volume_2 = volume_ellipsoid
        elif model_2 == "core-shell":
            formfactor_2 = formfactor_coreshell
    if distribution_type == "Double":
        return distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2
    elif distribution_type == "Single":
        return distribution_func_1,formfactor_1,volume_1
"""
likewise, get_variable_config returns the variable values currently recorded.
"""
@st.cache_data
def get_variable_config(dataset_values,global_model_state):
    distribution_type = global_model_state["distribution_type"]
    distribution_1 = global_model_state["distribution_1"]
    distribution_2 = global_model_state["distribution_2"]
    model_1 = global_model_state["model_1"]
    model_2 = global_model_state["model_2"]
    ###--------------------------------------
    #load parameter and option values
    ###--------------------------------------
    log_Ibg = dataset_values.get("log_Ibg")
    log_C = dataset_values.get("log_C")
    A_1 = dataset_values.get("A_1")
    Ibg = np.exp(log_Ibg)
    C = np.exp(log_C)
    combinedfactor_1 = np.exp(-A_1)
    Rm_1 = dataset_values.get("Rm_1")
    theta_list = [log_Ibg,log_C,A_1,Rm_1]
    #dummy values
    sigma_rm_1 = None
    sigma_rm_2 = None
    k_1 = None
    k_2 = None
    mu_1 = None
    mu_2 = None
    if distribution_1 != "mono":
        sigma_1 = dataset_values.get("Sigma_1")
        theta_list.append(sigma_1)
        sigma_rm_1 = sigma_1*Rm_1

    if model_1 == "ellipsoid":
        k_1 = dataset_values.get("kellipsoid_1")
        theta_list.append(k_1)
    elif model_1 == "core-shell":
        k_1 = dataset_values.get("kshell_1")
        theta_list.append(k_1)
        mu_1 = dataset_values.get("mu_1")
        theta_list.append(mu_1)
        
    if distribution_type == "Double": 
        A_2 = dataset_values.get("A_2")
        combinedfactor_2 = np.exp(-A_2)
        Rm_2 = dataset_values.get("Rm_2")
        theta_list.append(A_2)
        theta_list.append(Rm_2)
        if distribution_2 != "mono":
            sigma_2 = dataset_values.get("Sigma_2")
            sigma_rm_2 = sigma_2*Rm_2
            theta_list.append(sigma_2)

        if model_2 == "ellipsoid":
            k_2 = dataset_values.get("kellipsoid_2")
            theta_list.append(k_2)
        elif model_2 == "core-shell":
            k_2 = dataset_values.get("kshell_2")
            theta_list.append(k_2)
            mu_2 = dataset_values.get("mu_2")
            theta_list.append(mu_2)

    theta = np.array(theta_list)
    if distribution_type == "Double": 
        return Ibg,C,combinedfactor_1,Rm_1,sigma_rm_1,k_1,mu_1,combinedfactor_2,Rm_2,sigma_rm_2,k_2,mu_2,theta
    else:
        return Ibg,C,combinedfactor_1,Rm_1,sigma_rm_1,k_1,mu_1,theta
"""
get_BO_config: returns all possible data structures that might be used in subsequent analysis. 
Essentially a cleanup function that hides all complex logic
"""    
@st.cache_resource
def get_BO_config(settings_df,nuc_dataset_values,mag_dataset_values,global_model_state):
    distribution_type = global_model_state["distribution_type"]
    distribution_1 = global_model_state["distribution_1"]
    distribution_2 = global_model_state["distribution_2"]
    magnetic_scattering_background = global_model_state["magnetic_scattering_background"]
    model_1 = global_model_state["model_1"]
    model_2 = global_model_state["model_2"]
    #range settings import
    range_A = [settings_df.loc["min","A"],settings_df.loc["max","A"]] #combined factor
    range_Rm = [settings_df.loc["min","Rm"],settings_df.loc["max","Rm"]] 
    range_C = [settings_df.loc["min","log_C"],settings_df.loc["max","log_C"]] #approximation parameter
    range_Ibg = [settings_df.loc["min","log_Ibg"],settings_df.loc["max","log_Ibg"]]
    range_sigma = [settings_df.loc["min","Sigma"],settings_df.loc["max","Sigma"]]
    range_kellipsoid = [settings_df.loc["min","kellipsoid"],settings_df.loc["max","kellipsoid"]]
    range_kshell = [settings_df.loc["min","kshell"],settings_df.loc["max","kshell"]]
    range_mu = [settings_df.loc["min","mu"],settings_df.loc["max","mu"]]

    #user-related bound ranges
    search_width_A = settings_df.loc["searchwidth","A"]
    search_width_Rm = settings_df.loc["searchwidth","Rm"]
    search_width_C = settings_df.loc["searchwidth","log_C"]
    search_width_Ibg = settings_df.loc["searchwidth","log_Ibg"]
    search_width_sigma = settings_df.loc["searchwidth","Sigma"]
    search_width_kellipsoid = settings_df.loc["searchwidth","kellipsoid"]
    search_width_kshell = settings_df.loc["searchwidth","kshell"]
    search_width_mu = settings_df.loc["searchwidth","mu"]

    #first jointfit bound ranges
    search_width_A_jointfit = settings_df.loc["searchwidth_jointfit","A"]
    search_width_Rm_jointfit = settings_df.loc["searchwidth_jointfit","Rm"]
    search_width_C_jointfit = settings_df.loc["searchwidth_jointfit","log_C"]
    search_width_Ibg_jointfit = settings_df.loc["searchwidth_jointfit","log_Ibg"]
    search_width_sigma_jointfit = settings_df.loc["searchwidth_jointfit","Sigma"]
    search_width_kellipsoid_jointfit = settings_df.loc["searchwidth_jointfit","kellipsoid"]
    search_width_kshell_jointfit = settings_df.loc["searchwidth_jointfit","kshell"]
    search_width_mu_jointfit = settings_df.loc["searchwidth_jointfit","mu"]

    #pareto slender bound ranges
    search_width_A_jointfit_fine = settings_df.loc["searchwidth_fine","A"]
    search_width_Rm_jointfit_fine = settings_df.loc["searchwidth_fine","Rm"]
    search_width_C_jointfit_fine = settings_df.loc["searchwidth_fine","log_C"]
    search_width_Ibg_jointfit_fine = settings_df.loc["searchwidth_fine","log_Ibg"]
    search_width_sigma_jointfit_fine = settings_df.loc["searchwidth_fine","Sigma"]
    search_width_kellipsoid_jointfit_fine = settings_df.loc["searchwidth_fine","kellipsoid"]
    search_width_kshell_jointfit_fine = settings_df.loc["searchwidth_fine","kshell"]
    search_width_mu_jointfit_fine = settings_df.loc["searchwidth_fine","mu"]

    #value import
    log_Ibg_nuc = nuc_dataset_values.get("log_Ibg")
    log_C_nuc = nuc_dataset_values.get("log_C")
    A_1_nuc = nuc_dataset_values.get("A_1")
    A_2_nuc = nuc_dataset_values.get("A_2")
    Ibg_nuc = np.exp(log_Ibg_nuc)
    C_nuc = np.exp(log_C_nuc)
    combinedfactor_1_nuc = np.exp(-A_1_nuc)
    combinedfactor_2_nuc = np.exp(-A_2_nuc)
    Rm_1_nuc = nuc_dataset_values.get("Rm_1")
    sigma_1_nuc = nuc_dataset_values.get("Sigma_1")
    sigma_rm_1_nuc = sigma_1_nuc*Rm_1_nuc
    Rm_2_nuc = nuc_dataset_values.get("Rm_2")
    sigma_2_nuc = nuc_dataset_values.get("Sigma_2")
    sigma_rm_2_nuc = sigma_2_nuc*Rm_2_nuc
    k_1_nuc = nuc_dataset_values.get("kellipsoid_1")
    k_1_nuc_cs = nuc_dataset_values.get("kshell_1")
    k_2_nuc = nuc_dataset_values.get("kellipsoid_2")
    k_2_nuc_cs = nuc_dataset_values.get("kshell_2")
    mu_1_nuc = nuc_dataset_values.get("mu_1")
    mu_2_nuc = nuc_dataset_values.get("mu_2")

    log_Ibg_mag = mag_dataset_values.get("log_Ibg")
    log_C_mag = mag_dataset_values.get("log_C")
    A_1_mag = mag_dataset_values.get("A_1")
    A_2_mag = mag_dataset_values.get("A_2")
    Ibg_mag = np.exp(log_Ibg_mag)
    C_mag = np.exp(log_C_mag)
    combinedfactor_1_mag = np.exp(-A_1_mag)
    combinedfactor_2_mag = np.exp(-A_2_mag)
    Rm_1_mag = mag_dataset_values.get("Rm_1")
    sigma_1_mag = mag_dataset_values.get("Sigma_1")
    sigma_rm_1_mag = sigma_1_mag*Rm_1_mag
    Rm_2_mag = mag_dataset_values.get("Rm_2")
    sigma_2_mag = mag_dataset_values.get("Sigma_2")
    sigma_rm_2_mag = sigma_2_mag*Rm_2_mag
    k_1_mag = mag_dataset_values.get("kellipsoid_1")
    k_1_mag_cs = mag_dataset_values.get("kshell_1")
    k_2_mag = mag_dataset_values.get("kellipsoid_2")
    k_2_mag_cs = mag_dataset_values.get("kshell_2")
    mu_1_mag = mag_dataset_values.get("mu_1")
    mu_2_mag = mag_dataset_values.get("mu_2")

    if magnetic_scattering_background == "Off":
        log_Ibg_mag = None
    num_parameters = 0
    #case by case initialization
    if distribution_type == "Single":
        if (distribution_1 == "normal") or (distribution_1 == "log_normal") :
            if model_1 == "sphere":
                num_parameters_1 = 5
            elif model_1 == "ellipsoid":
                num_parameters_1 = 6
            elif model_1 == "core-shell":
                num_parameters_1 = 7
            num_parameters = num_parameters_1
        elif distribution_1 == "mono":
            if model_1 == "sphere":
                num_parameters_1 = 4
            elif model_1 == "ellipsoid":
                num_parameters_1 = 5
            elif model_1 == "core-shell":
                num_parameters_1 = 6
            num_parameters = num_parameters_1
    elif distribution_type == "Double":
        if (distribution_1 == "normal") or (distribution_1 == "log_normal") :
            if model_1 == "sphere":
                num_parameters_1 = 5
            elif model_1 == "ellipsoid":
                num_parameters_1 = 6
            elif model_1 == "core-shell":
                num_parameters_1 = 7
        elif distribution_1 == "mono":
            if model_1 == "sphere":
                num_parameters_1 = 4
            elif model_1 == "ellipsoid":
                num_parameters_1 = 5
            elif model_1 == "core-shell":
                num_parameters_1 = 6
        if (distribution_2 == "normal") or (distribution_2 == "log_normal") :
            if model_2 == "sphere":
                num_parameters_2 = 3
            elif model_2 == "ellipsoid":
                num_parameters_2 = 4
            elif model_2 == "core-shell":
                num_parameters_2 = 5
        elif distribution_2 == "mono":
            if model_2 == "sphere":
                num_parameters_2 = 2
            elif model_2 == "ellipsoid":
                num_parameters_2 = 3
            elif model_2 == "core-shell":
                num_parameters_2 = 4
        num_parameters = num_parameters_1+num_parameters_2
    
    bounds_1 = torch.tensor([[max(range_Ibg[0],log_Ibg_nuc-search_width_Ibg),min(range_Ibg[1],log_Ibg_nuc+search_width_Ibg)],
                            [max(range_C[0],log_C_nuc-search_width_C),min(range_C[1],log_C_nuc+search_width_C)],
                            [max(range_A[0],A_1_nuc-search_width_A),min(range_A[1],A_1_nuc+search_width_A)],
                            [max(range_Rm[0],Rm_1_nuc-search_width_Rm),min(range_Rm[1],Rm_1_nuc+search_width_Rm)]])
    bounds_box_1 = [range_Ibg,range_C,range_A,range_Rm]
    joint_searchwidth_1 = [search_width_Ibg_jointfit,search_width_C_jointfit,search_width_A_jointfit,search_width_Rm_jointfit]
    joint_searchwidth_1_fine = [search_width_Ibg_jointfit_fine,search_width_C_jointfit_fine,search_width_A_jointfit_fine,search_width_Rm_jointfit_fine]
    bounds_2 = torch.tensor([[max(range_A[0],A_2_nuc-search_width_A),min(range_A[1],A_2_nuc+search_width_A)],
                            [max(range_Rm[0],Rm_2_nuc-search_width_Rm),min(range_Rm[1],Rm_2_nuc+search_width_Rm)]])
    bounds_box_2 = [range_A,range_Rm]
    joint_searchwidth_2 = [search_width_A_jointfit,search_width_Rm_jointfit]
    joint_searchwidth_2_fine = [search_width_A_jointfit_fine,search_width_Rm_jointfit_fine]
    #mag signal bounds
    if log_Ibg_mag is not None:
        bounds_1_mag = torch.tensor([[max(range_Ibg[0],log_Ibg_mag-search_width_Ibg),min(range_Ibg[1],log_Ibg_mag+search_width_Ibg)],
                                [max(range_C[0],log_C_mag-search_width_C),min(range_C[1],log_C_mag+search_width_C)],
                                [max(range_A[0],A_1_mag-search_width_A),min(range_A[1],A_1_mag+search_width_A)],
                                [max(range_Rm[0],Rm_1_mag-search_width_Rm),min(range_Rm[1],Rm_1_mag+search_width_Rm)]])
        params_mag_1 = ['background_log_mag','C_log_mag','combinedfactor_log_mag']
    else:
        bounds_1_mag = torch.tensor([[max(range_A[0],A_1_mag-search_width_A),min(range_A[1],A_1_mag+search_width_A)],
                                [max(range_Rm[0],Rm_1_mag-search_width_Rm),min(range_Rm[1],Rm_1_mag+search_width_Rm)]])
        params_mag_1 = ['combinedfactor_log_mag']
    bounds_2_mag = torch.tensor([[max(range_A[0],A_2_mag-search_width_A),min(range_A[1],A_2_mag+search_width_A)],
                            [max(range_Rm[0],Rm_2_mag-search_width_Rm),min(range_Rm[1],Rm_2_mag+search_width_Rm)]])

    #list of parameters to optimize
    params_1 = ['background_log','C_log','combinedfactor_log','R']
    params_2 = ['combinedfactor_2_log','R_2']
    common_params = ['R']
    possible_common_params_single = ['R','sigma','k']
    possible_common_params_double = ['R','sigma','k','R_2','sigma_2','k_2']
    #list of fixed parameters
    fixed_kwargs = {'approx': I_porod}
    # assign distribution function
    distribution_func_1 = "mono"
    distribution_func_2 = "mono"
    if distribution_type == "Single":
        if distribution_1 == "log_normal":
            distribution_func_1 = distribution_lognormal
            bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1_nuc-search_width_sigma),min(range_sigma[1],sigma_1_nuc+search_width_sigma)]])])
            bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
            params_1.append('sigma')
            common_params.append('sigma')
            bounds_box_1.append(range_sigma)
            joint_searchwidth_1.append(search_width_sigma_jointfit)
            joint_searchwidth_1_fine.append(search_width_sigma_jointfit_fine)
        elif distribution_1 == "normal":
            distribution_func_1 = distribution_normal
            bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1_nuc-search_width_sigma),min(range_sigma[1],sigma_1_nuc+search_width_sigma)]])])
            bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
            params_1.append('sigma')
            common_params.append('sigma')
            bounds_box_1.append(range_sigma)
            joint_searchwidth_1.append(search_width_sigma_jointfit)
            joint_searchwidth_1_fine.append(search_width_sigma_jointfit_fine)
    elif distribution_type == "Double":
        common_params.append('R_2')
        if distribution_1 == "log_normal":
            distribution_func_1 = distribution_lognormal
            bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1_nuc-search_width_sigma),min(range_sigma[1],sigma_1_nuc+search_width_sigma)]])])
            bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
            params_1.append('sigma')
            common_params.append('sigma')
            bounds_box_1.append(range_sigma)
            joint_searchwidth_1.append(search_width_sigma_jointfit)
            joint_searchwidth_1_fine.append(search_width_sigma_jointfit_fine)
        elif distribution_1 == "normal":
            distribution_func_1 = distribution_normal
            bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1_nuc-search_width_sigma),min(range_sigma[1],sigma_1_nuc+search_width_sigma)]])])
            bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
            params_1.append('sigma')
            common_params.append('sigma')
            bounds_box_1.append(range_sigma)
            joint_searchwidth_1.append(search_width_sigma_jointfit)
            joint_searchwidth_1_fine.append(search_width_sigma_jointfit_fine)
        if distribution_2 == "log_normal":
            distribution_func_2 = distribution_lognormal
            bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_sigma[0],sigma_2_nuc-search_width_sigma),min(range_sigma[1],sigma_2_nuc+search_width_sigma)]])])
            bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_sigma[0],sigma_2_mag-search_width_sigma),min(range_sigma[1],sigma_2_mag+search_width_sigma)]])])
            params_2.append('sigma_2')
            common_params.append('sigma_2')
            bounds_box_2.append(range_sigma)
            joint_searchwidth_2.append(search_width_sigma_jointfit)
            joint_searchwidth_2_fine.append(search_width_sigma_jointfit_fine)
        elif distribution_2 == "normal":
            distribution_func_2 = distribution_normal
            bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_sigma[0],sigma_2_nuc-search_width_sigma),min(range_sigma[1],sigma_2_nuc+search_width_sigma)]])])
            bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_sigma[0],sigma_2_mag-search_width_sigma),min(range_sigma[1],sigma_2_mag+search_width_sigma)]])])
            params_2.append('sigma_2')
            common_params.append('sigma_2')
            bounds_box_2.append(range_sigma)
            joint_searchwidth_2.append(search_width_sigma_jointfit)
            joint_searchwidth_2_fine.append(search_width_sigma_jointfit_fine)

    #assign models
    if model_1 == "sphere":
        formfactor_1 = formfactor_sphere
        volume_1 = volume_sphere
    elif model_1 == "ellipsoid":
        formfactor_1 = formfactor_ellipsoid
        volume_1 = volume_ellipsoid
        bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_kellipsoid[0],k_1_nuc-search_width_kellipsoid),min(range_kellipsoid[1],k_1_nuc+search_width_kellipsoid)]])])
        bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_kellipsoid[0],k_1_mag-search_width_kellipsoid),min(range_kellipsoid[1],k_1_mag+search_width_kellipsoid)]])])
        params_1.append('k')
        common_params.append('k')
        bounds_box_1.append(range_kellipsoid)
        joint_searchwidth_1.append(search_width_kellipsoid_jointfit)
        joint_searchwidth_1_fine.append(search_width_kellipsoid_jointfit_fine)
    elif model_1 == "core-shell":
        formfactor_1 = formfactor_coreshell
        volume_1 = volume_sphere
        bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_kshell[0],k_1_nuc_cs-search_width_kshell),min(range_kshell[1],k_1_nuc_cs+search_width_kshell)]])])
        bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_mu[0],mu_1_nuc-search_width_mu),min(range_mu[1],mu_1_nuc+search_width_mu)]])])
        bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_kshell[0],k_1_mag_cs-search_width_kshell),min(range_kshell[1],k_1_mag_cs+search_width_kshell)]])])
        bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_mu[0],mu_1_mag-search_width_mu),min(range_mu[1],mu_1_mag+search_width_mu)]])])
        params_1.append('k')
        common_params.append('k')
        bounds_box_1.append(range_kshell)
        joint_searchwidth_1.append(search_width_kshell_jointfit)
        joint_searchwidth_1_fine.append(search_width_kshell_jointfit_fine)
        params_1.append('mu')
        bounds_box_1.append(range_mu)
        joint_searchwidth_1.append(search_width_mu_jointfit)
        joint_searchwidth_1_fine.append(search_width_mu_jointfit_fine)


    if model_2 == "sphere":
        formfactor_2 = formfactor_sphere
        volume_2 = volume_sphere
    elif model_2 == "ellipsoid":
        formfactor_2 = formfactor_ellipsoid
        volume_2 = volume_ellipsoid
        bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_kellipsoid[0],k_2_nuc-search_width_kellipsoid),min(range_kellipsoid[1],k_2_nuc+search_width_kellipsoid)]])])
        bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_kellipsoid[0],k_2_mag-search_width_kellipsoid),min(range_kellipsoid[1],k_2_mag+search_width_kellipsoid)]])])
        params_2.append('k_2')
        common_params.append('k_2')
        bounds_box_2.append(range_kellipsoid)
        joint_searchwidth_2.append(search_width_kellipsoid_jointfit)
        joint_searchwidth_2_fine.append(search_width_kellipsoid_jointfit_fine)
    elif model_2 == "core-shell":
        formfactor_2 = formfactor_coreshell
        volume_2 = volume_sphere
        bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_kshell[0],k_2_nuc_cs-search_width_kshell),min(range_kshell[1],k_2_nuc_cs+search_width_kshell)]])])
        bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_mu[0],mu_2_nuc-search_width_mu),min(range_mu[1],mu_2_nuc+search_width_mu)]])])
        bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_kshell[0],k_2_mag_cs-search_width_kshell),min(range_kshell[1],k_2_mag_cs+search_width_kshell)]])])
        bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_mu[0],mu_2_mag-search_width_mu),min(range_mu[1],mu_2_mag+search_width_mu)]])])
        params_2.append('k_2')
        common_params.append('k_2')
        bounds_box_2.append(range_kshell)
        joint_searchwidth_2.append(search_width_kshell_jointfit)
        joint_searchwidth_2_fine.append(search_width_kshell_jointfit_fine)
        params_2.append('mu_2')
        bounds_box_2.append(range_mu)
        joint_searchwidth_2.append(search_width_mu_jointfit)
        joint_searchwidth_2_fine.append(search_width_mu_jointfit_fine)

    fixed_kwargs['distribution'] = distribution_func_1
    fixed_kwargs['formfactor'] = formfactor_1
    fixed_kwargs['volume'] = volume_1
    if distribution_type == "Double":
        fixed_kwargs['distribution_2'] = distribution_func_2
        fixed_kwargs['formfactor_2'] = formfactor_2
        fixed_kwargs['volume_2'] = volume_2
    pos_array_A_2 = -1
    pos_array_mu_1 = -1
    pos_array_mu_2 = -1
    if log_Ibg_mag is not None:
        nuc_pos_list = [0,1,2] #corresponds to Ibg, C, A, or only A if porod and bg is disabled
        mag_pos_list = [0,1,2]
    else:
        nuc_pos_list = [2]
        mag_pos_list = [0]
    #assign BO bounds
    if distribution_type == "Double":
        bounds = torch.cat([bounds_1,bounds_2])
        bounds_mag = torch.cat([bounds_1_mag,bounds_2_mag])
        bounds = bounds.T
        bounds_mag = bounds_mag.T
        params = params_1 + params_2
        bounds_box_np_1 = np.array(bounds_box_1)
        joint_searchwidth = joint_searchwidth_1 + joint_searchwidth_2
        joint_searchwidth_fine = joint_searchwidth_1_fine + joint_searchwidth_2_fine
        bounds_box_np_2 = np.array(bounds_box_2)
        pos_array_A_2 = bounds_box_np_1.shape[0] # notes relative position of combined factor 1 in parameter set
        nuc_pos_list.append(pos_array_A_2)
        mag_pos_list.append(mag_pos_list[-1]+1)
        params_mag_1.append('combinedfactor_2_log_mag')
        if model_1 == "core-shell":
            pos_array_mu_1 = bounds_box_np_1.shape[0]-1 
            nuc_pos_list.append(pos_array_mu_1)
            mag_pos_list.append(mag_pos_list[-1]+1)
            params_mag_1.append('mu_mag')
        if model_2 == "core-shell":
            pos_array_mu_2 = bounds_box_np_1.shape[0] + bounds_box_np_2.shape[0]-1
            nuc_pos_list.append(pos_array_mu_2)
            mag_pos_list.append(mag_pos_list[-1]+1)
            params_mag_1.append('mu_2_mag')
        bounds_box_np = np.vstack([bounds_box_np_1,bounds_box_np_2])
    else:
        bounds = bounds_1.T
        bounds_mag = bounds_1_mag.T
        params = params_1
        bounds_box_np = np.array(bounds_box_1)
        joint_searchwidth = joint_searchwidth_1
        joint_searchwidth_fine = joint_searchwidth_1_fine
        if model_1 == "core-shell":
            pos_array_mu_1 = bounds_box_np.shape[0]-1
            nuc_pos_list.append(pos_array_mu_1)
            mag_pos_list.append(mag_pos_list[-1]+1)
            params_mag_1.append('mu_mag')
    if log_Ibg_mag is not None:
        params_firstfit_mag = params
    else:
        params_firstfit_mag = copy.deepcopy(params)
        params_firstfit_mag.pop(0)
        params_firstfit_mag.pop(0)
    params_mag = params_mag_1 + common_params
    jointfit_params = params + params_mag_1
    nuc_params_temp = np.array(params)
    mag_params_temp = np.array(params_mag)
    mask_params_nuc = ~np.isin(nuc_params_temp,common_params)
    mask_params_nuc_torch = torch.tensor(mask_params_nuc)
    mask_params_mag = ~np.isin(mag_params_temp,common_params)
    mask_params_mag_torch = torch.tensor(mask_params_mag)
    nuc_params_filtered = nuc_params_temp[mask_params_nuc].tolist()
    bounds_nuc_filtered = bounds[:,mask_params_nuc_torch]
    mag_params_filtered = mag_params_temp[mask_params_mag].tolist()
    bounds_mag_filtered = bounds_mag[:,mask_params_mag_torch]

    mask_params_nuc_del = np.isin(nuc_params_temp,common_params)
    mask_params_mag_del = np.isin(mag_params_temp,common_params)
    deleted_nuc_params = nuc_params_temp[mask_params_nuc_del]
    deleted_mag_params = mag_params_temp[mask_params_mag_del]

    return (bounds, 
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
            pos_array_mu_2,)