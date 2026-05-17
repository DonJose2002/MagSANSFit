import streamlit as st
from app.prototypes.variable import Variable
#def get_active_variables(model_state,dataset_name):
#
#    model = model_state["model"]
#
#    if model == "Sine":
#
#        return [
#
#    make_variable(
#        "amplitude",
#        dataset_name
#    ),
#
#    make_variable(
#        "frequency",
#        dataset_name
#    ),
#
#    make_variable(
#        "offset",
#        dataset_name
#    ),
#]
#
#    elif model == "Gaussian":
#
#        return [
#
#            make_variable(
#        "amplitude",
#        dataset_name
#    ),
#
#    make_variable(
#        "sigma",
#        dataset_name
#    ),
#
#    make_variable(
#        "offset",
#        dataset_name
#    ),
#        ]
#
#    return []

def update_variable_activity(dataset):

    variables = dataset["variables"]

    # -----------------------------------
    # FIRST DISABLE EVERYTHING
    # -----------------------------------

    for variable in variables.values():

        variable.active = False

    # -----------------------------------
    # MODEL LOGIC
    # -----------------------------------
    dataset_name = st.session_state.active_dataset
    distribution_type = st.session_state.global_model_state["distribution_type"]
    distribution_1 = st.session_state.global_model_state["distribution_1"]
    distribution_2 = st.session_state.global_model_state["distribution_2"]
    model_1 = st.session_state.global_model_state["model_1"]
    model_2 = st.session_state.global_model_state["model_2"]

    #analysis_mode =  st.session_state.global_model_state["analysis_mode"]
    magnetic_scattering_background = st.session_state.global_model_state["magnetic_scattering_background"]
    variables["log_Ibg"].active = True
    variables["log_C"].active = True
    variables["A_1"].active = True
    variables["Rm_1"].active = True
    if dataset_name == "Magnetic Signal":
        if magnetic_scattering_background == "Off":
            variables["log_Ibg"].active = False
            variables["log_C"].active = False

    if distribution_1 != "mono":
        variables["Sigma_1"].active = True
    if model_1 == "ellipsoid":
        variables["kellipsoid_1"].active = True
    if model_1 == "core-shell":
        variables["kshell_1"].active = True
        variables["mu_1"].active = True
    if distribution_type == "Double": 
        variables["A_2"].active = True
        variables["Rm_2"].active = True       
        if distribution_2 != "mono":
            variables["Sigma_2"].active = True
        if model_2 == "ellipsoid":
            variables["kellipsoid_2"].active = True
        if model_2 == "core-shell":
            variables["kshell_2"].active = True
            variables["mu_2"].active = True
        