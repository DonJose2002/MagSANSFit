import streamlit as st
import plotly.graph_objects as go
import numpy as np
import copy

from app.utilities.cache_generator import get_model_config
from problems.coreshell import formfactor_coreshell
from problems.distributions import distribution_normal,distribution_lognormal
from problems.ellipsoid import volume_ellipsoid,formfactor_ellipsoid
from problems.sphere import volume_sphere,formfactor_sphere
from problems.approximations import I_porod
from core.models.assembled_problem import final_intensity_spheroid,nano_intensity_spheroid
from core.models.loss_functions import loss_chi2
from core.models.single_loss import chi2_final_intensity_spheroid_double,chi2_final_intensity_spheroid_single,chi2_nano_intensity_spheroid_double,chi2_nano_intensity_spheroid_single
def create_plot(treated_input):
    ###--------------------------------------
    #Gather info from session_state
    ###--------------------------------------
    dataset_nuc = st.session_state.datasets["Nuclear Signal"]
    dataset = st.session_state.datasets["Magnetic Signal"]
    values_nuc = dataset_nuc["values"]
    values_mag = dataset["values"]
    dataset_name = st.session_state.active_dataset
    distribution_type = st.session_state.global_model_state["distribution_type"]
    distribution_1 = st.session_state.global_model_state["distribution_1"]
    distribution_2 = st.session_state.global_model_state["distribution_2"]
    model_1 = st.session_state.global_model_state["model_1"]
    model_2 = st.session_state.global_model_state["model_2"]
    analysis_mode =  st.session_state.global_model_state["analysis_mode"]
    magnetic_scattering_background = st.session_state.global_model_state["magnetic_scattering_background"]
    ###--------------------------------------
    #load parameter and option values
    ###--------------------------------------
    log_Ibg_nuc = values_nuc.get("log_Ibg")
    log_C_nuc = values_nuc.get("log_C")
    A_1_nuc = values_nuc.get("A_1")
    Ibg_nuc = np.exp(log_Ibg_nuc)
    C_nuc = np.exp(log_C_nuc)
    combinedfactor_1_nuc = np.exp(-A_1_nuc)
    Rm_1_nuc = values_nuc.get("Rm_1")
    theta_nuc_list = [log_Ibg_nuc,log_C_nuc,A_1_nuc,Rm_1_nuc]
    distribution_func_1 = "mono"
    formfactor_1 = formfactor_sphere
    volume_1 = volume_sphere
    #dummy values
    sigma_rm_1_nuc = None
    sigma_rm_2_nuc = None
    k_1_nuc = None
    k_2_nuc = None
    mu_1_nuc = None
    mu_2_nuc = None
    if distribution_1 != "mono":
        sigma_1_nuc = values_nuc.get("Sigma_1")
        theta_nuc_list.append(sigma_1_nuc)
        sigma_rm_1_nuc = sigma_1_nuc*Rm_1_nuc

        if distribution_1 == "log_normal":
            distribution_func_1 = distribution_lognormal
        elif distribution_1 =="normal":
            distribution_func_1 = distribution_normal

    if model_1 == "ellipsoid":
        formfactor_1 = formfactor_ellipsoid
        volume_1 = volume_ellipsoid
        k_1_nuc = values_nuc.get("kellipsoid_1")
        theta_nuc_list.append(k_1_nuc)
    elif model_1 == "core-shell":
        formfactor_1 = formfactor_coreshell
        k_1_nuc = values_nuc.get("kshell_1")
        theta_nuc_list.append(k_1_nuc)
        mu_1_nuc = values_nuc.get("mu_1")
        theta_nuc_list.append(mu_1_nuc)
        
    if distribution_type == "Double": 
        A_2_nuc = values_nuc.get("A_2")
        combinedfactor_2_nuc = np.exp(-A_2_nuc)
        Rm_2_nuc = values_nuc.get("Rm_2")
        theta_nuc_list.append(A_2_nuc)
        theta_nuc_list.append(Rm_2_nuc)
        distribution_func_2 = "mono"
        formfactor_2 = formfactor_sphere
        volume_2 = volume_sphere
        if distribution_2 != "mono":
            sigma_2_nuc = values_nuc.get("Sigma_2")
            sigma_rm_2_nuc = sigma_2_nuc*Rm_2_nuc
            theta_nuc_list.append(sigma_2_nuc)

            if distribution_2 == "log_normal":
                distribution_func_2 = distribution_lognormal
            elif distribution_2 =="normal":
                distribution_func_2 = distribution_normal

        if model_2 == "ellipsoid":
            formfactor_2 = formfactor_ellipsoid
            volume_2 = volume_ellipsoid
            k_2_nuc = values_nuc.get("kellipsoid_2")
            theta_nuc_list.append(k_2_nuc)
        elif model_2 == "core-shell":
            formfactor_2 = formfactor_coreshell
            k_2_nuc = values_nuc.get("kshell_2")
            theta_nuc_list.append(k_2_nuc)
            mu_2_nuc = values_nuc.get("mu_2")
            theta_nuc_list.append(mu_2_nuc)

    theta_nuc = np.array(theta_nuc_list)
    #print(theta_nuc)
    #print(len(theta_nuc))

    if analysis_mode == "Distribution Fitting":
        test_Q, test_Inuc, test_sigmaInuc, test_sigmaQ = treated_input
    elif analysis_mode == "Nuclear-Magnetic Joint Analysis":
        test_Q, test_Inuc, test_sigmaInuc, test_sigmaQ, test_Imag, test_sigmaImag = treated_input
        A_1_mag = values_mag.get("A_1")
        Rm_1_mag = values_mag.get("Rm_1")
        
        combinedfactor_1_mag = np.exp(-A_1_mag)
        if magnetic_scattering_background == "On":
            log_Ibg_mag = values_mag.get("log_Ibg")
            log_C_mag = values_mag.get("log_C")
            Ibg_mag = np.exp(log_Ibg_mag)
            C_mag = np.exp(log_C_mag)
            theta_mag_list = [log_Ibg_mag,log_C_mag,A_1_mag,Rm_1_mag]
        elif magnetic_scattering_background == "Off":
            log_Ibg_mag = None
            log_C_mag = None
            theta_mag_list = [A_1_mag,Rm_1_mag]
        sigma_rm_1_mag = None
        sigma_rm_2_mag = None
        k_1_mag = None
        k_2_mag = None
        mu_1_mag = None
        mu_2_mag = None
        if distribution_1 != "mono":
            sigma_1_mag = values_mag.get("Sigma_1")
            theta_mag_list.append(sigma_1_mag)
            sigma_rm_1_mag = sigma_1_mag*Rm_1_mag

        if model_1 == "ellipsoid":
            k_1_mag = values_mag.get("kellipsoid_1")
            theta_mag_list.append(k_1_mag)
        elif model_1 == "core-shell":
            k_1_mag = values_mag.get("kshell_1")
            theta_mag_list.append(k_1_mag)
            mu_1_mag = values_mag.get("mu_1")
            theta_mag_list.append(mu_1_mag)
            
        if distribution_type == "Double": 
            A_2_mag = values_mag.get("A_2")
            combinedfactor_2_mag = np.exp(-A_2_mag)
            Rm_2_mag = values_mag.get("Rm_2")
            theta_mag_list.append(A_2_mag)
            theta_mag_list.append(Rm_2_mag)
            if distribution_2 != "mono":
                sigma_2_mag = values_mag.get("Sigma_2")
                theta_mag_list.append(sigma_2_mag)
                sigma_rm_2_mag = sigma_2_mag*Rm_2_mag

            if model_2 == "ellipsoid":
                k_2_mag = values_mag.get("kellipsoid_2")
                theta_mag_list.append(k_2_mag)
            elif model_2 == "core-shell":
                k_2_mag = values_mag.get("kshell_2")
                theta_mag_list.append(k_2_mag)
                mu_2_mag = values_mag.get("mu_2")
                theta_mag_list.append(mu_2_mag)
        theta_mag = np.array(theta_mag_list)
    model_Q = np.linspace(max(np.min(test_Q)-0.1,0.01),np.max(test_Q)+0.1,100)
    if distribution_type == "Double":
        model_Inuc = [(final_intensity_spheroid(
            I_porod,combinedfactor_1_nuc,distribution_func_1,formfactor_1,volume_1,
            Ibg_nuc,q,Rm_1_nuc,C_nuc,sigma=sigma_rm_1_nuc,k=k_1_nuc,mu=mu_1_nuc
            )+nano_intensity_spheroid(combinedfactor_2_nuc,distribution_func_2,formfactor_2,volume_2,
                                      q,Rm_2_nuc,sigma=sigma_rm_2_nuc,k=k_2_nuc,mu=mu_2_nuc)) for q in model_Q]
        chi2_nuc = chi2_final_intensity_spheroid_double(theta_nuc,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Inuc,test_sigmaInuc)
    elif distribution_type == "Single":
        model_Inuc = [final_intensity_spheroid(
            I_porod,combinedfactor_1_nuc,distribution_func_1,formfactor_1,volume_1,
            Ibg_nuc,q,Rm_1_nuc,C_nuc,sigma=sigma_rm_1_nuc,k=k_1_nuc,mu=mu_1_nuc
            ) for q in model_Q]
        #print(theta_nuc)
        #print(len(theta_nuc))
        #print(distribution_1)
        #print(distribution_func_1)
        chi2_nuc = chi2_final_intensity_spheroid_single(theta_nuc,distribution_func_1,formfactor_1,volume_1,test_Q,test_Inuc,test_sigmaInuc)
    #print(theta_nuc)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = test_Q,
        y = np.log(test_Inuc),
        mode = "markers",
        name = "Experimental Nuclear Scattering",
        marker = dict(size=10),
        visible=(dataset_name=="Nuclear Signal")
    ))
    fig.add_trace(go.Scatter(
        x=model_Q,
        y=np.log(model_Inuc),
        mode="lines",
        name = "Model Nuclear Scattering",
        visible=(dataset_name=="Nuclear Signal")
    ))
    chi2_mag = None
    if analysis_mode == "Nuclear-Magnetic Joint Analysis":
        if distribution_type == "Double":
            if log_Ibg_mag is not None:
                model_Imag = [(final_intensity_spheroid(
                    I_porod,combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                    Ibg_mag,q,Rm_1_mag,C_mag,sigma=sigma_rm_1_mag,k=k_1_mag,mu=mu_1_mag
                    )+nano_intensity_spheroid(combinedfactor_2_mag,distribution_func_2,formfactor_2,volume_2,
                                            q,Rm_2_mag,sigma=sigma_rm_2_mag,k=k_2_mag,mu=mu_2_mag)) for q in model_Q]
                chi2_mag = chi2_final_intensity_spheroid_double(theta_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                model_Imag = [(nano_intensity_spheroid(
                    combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                    q,Rm_1_mag,sigma=sigma_rm_1_mag,k=k_1_mag,mu=mu_1_mag
                    )+nano_intensity_spheroid(combinedfactor_2_mag,distribution_func_2,formfactor_2,volume_2,
                                            q,Rm_2_mag,sigma=sigma_rm_2_mag,k=k_2_mag,mu=mu_2_mag)) for q in model_Q]
                chi2_mag = chi2_nano_intensity_spheroid_double(theta_mag,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        elif distribution_type == "Single":
            if log_Ibg_mag is not None:
                model_Imag = [final_intensity_spheroid(
                    I_porod,combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                    Ibg_mag,q,Rm_1_mag,C_mag,sigma=sigma_rm_1_mag,k=k_1_mag,mu=mu_1_mag
                    ) for q in model_Q]
                chi2_mag = chi2_final_intensity_spheroid_single(theta_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            else:
                
                model_Imag = [nano_intensity_spheroid(
                    combinedfactor_1_mag,distribution_func_1,formfactor_1,volume_1,
                    q,Rm_1_mag,sigma=sigma_rm_1_mag,k=k_1_mag,mu=mu_1_mag
                    ) for q in model_Q]
                chi2_mag = chi2_nano_intensity_spheroid_single(theta_mag,distribution_func_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        fig.add_trace(go.Scatter(
            x = test_Q,
            y = np.log(test_Imag),
            mode = "markers",
            name = "Experimental Magnetic Scattering",
            marker = dict(size=10),
            visible=(dataset_name=="Magnetic Signal")
        ))
        fig.add_trace(go.Scatter(
            x=model_Q,
            y=np.log(model_Imag),
            mode="lines",
            name = "Model Magnetic Scattering",
            visible=(dataset_name=="Magnetic Signal")
        ))

    fig.update_layout(
    xaxis_title = "Q(nm-1)",
    yaxis_title = "log(I(Q))(cm-1)",
    template = "plotly_white"
)
    
    return fig,chi2_nuc,chi2_mag
"""
single_plot: returns plotly figure object based on read file, model_state and variable_values.
uses cache_resource to avoid recalculation when switching nuc/mag dataset
"""
@st.cache_resource
def single_plot(treated_input, global_model_state, active_dataset, variable_values, magnetic_scattering_background): #magnetic_scattering_background is input to ensure proper response during toggling
    ### gather st info ###
    analysis_mode =  global_model_state["analysis_mode"]
    distribution_type = global_model_state["distribution_type"]
    model_state = get_model_config(global_model_state)
    ### input decode ###
    if distribution_type == "Double":
        distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2 = model_state
        Ibg,C,combinedfactor_1,Rm_1,sigma_rm_1,k_1,mu_1,combinedfactor_2,Rm_2,sigma_rm_2,k_2,mu_2,theta = variable_values
    elif distribution_type == "Single":
        distribution_func_1,formfactor_1,volume_1 = model_state
        Ibg,C,combinedfactor_1,Rm_1,sigma_rm_1,k_1,mu_1,theta = variable_values
        
    theta_internal = theta.copy()
    if active_dataset == "Magnetic Signal" and magnetic_scattering_background == "Off":
        Ibg = None
        C = None
        theta_internal = theta[2:]
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
    ### calculate curve and generate fig object ###
    model_Q = np.linspace(max(np.min(test_Q)-0.1,0.01),np.max(test_Q)+0.1,100)
    if distribution_type == "Double":
        if Ibg is not None:
            model_I = [(final_intensity_spheroid(
                    I_porod,combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                    Ibg,q,Rm_1,C,sigma=sigma_rm_1,k=k_1,mu=mu_1
                    )+nano_intensity_spheroid(combinedfactor_2,distribution_func_2,formfactor_2,volume_2,
                                            q,Rm_2,sigma=sigma_rm_2,k=k_2,mu=mu_2)) for q in model_Q]
            chi2 = chi2_final_intensity_spheroid_double(theta_internal,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
        else:
            model_I = [(nano_intensity_spheroid(
                    combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                    q,Rm_1,sigma=sigma_rm_1,k=k_1,mu=mu_1
                    )+nano_intensity_spheroid(combinedfactor_2,distribution_func_2,formfactor_2,volume_2,
                                            q,Rm_2,sigma=sigma_rm_2,k=k_2,mu=mu_2)) for q in model_Q]
            chi2 = chi2_nano_intensity_spheroid_double(theta_internal,distribution_func_1,distribution_func_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)  
    elif distribution_type == "Single":
        if Ibg is not None:
            model_I = [final_intensity_spheroid(
                    I_porod,combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                    Ibg,q,Rm_1,C,sigma=sigma_rm_1,k=k_1,mu=mu_1
                    ) for q in model_Q]
            #print(theta_internal)
            chi2 = chi2_final_intensity_spheroid_single(theta_internal,distribution_func_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
        else:
                
            model_I = [nano_intensity_spheroid(
                    combinedfactor_1,distribution_func_1,formfactor_1,volume_1,
                    q,Rm_1,sigma=sigma_rm_1,k=k_1,mu=mu_1
                    ) for q in model_Q]
            chi2 = chi2_nano_intensity_spheroid_single(theta_internal,distribution_func_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x = test_Q,
            y = np.log(test_I),
            mode = "markers",
            name = "Experimental Scattering",
            marker = dict(size=5)
        ))
    fig.add_trace(go.Scatter(
            x=model_Q,
            y=np.log(model_I),
            mode="lines",
            name = "Model Scattering"
        ))

    fig.update_layout(
    xaxis_title = "Q(nm-1)",
    yaxis_title = "log(I(Q))(cm-1)",
    template = "plotly_white"
)
    return fig,chi2