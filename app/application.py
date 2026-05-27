import sys
import os

sys.path.append(os.getcwd())

import streamlit as st

from app.utilities.state import initialize_state,initialize_variable_values
from app.utilities.visibility import update_variable_activity
from app.utilities.validation import validate_all_variable_values
from app.utilities.cache_generator import get_BO_config, get_model_config, get_variable_config
from app.components.single_analysis import singular_fit
from app.components.joint_analysis import joint_analysis
from app.components.controls import render_controls
from app.components.model_selector import render_model_selector
from app.components.plotting import single_plot
from app.components.settings_editor import render_settings_editor
from app.components.preset_manager import save_preset,load_preset
from app.components.preset_manager import apply_pending_preset
from app.components.data_loader import render_data_loader
from app.components.misc_controls import render_misc_controls
st.set_page_config(layout="wide")

st.title("VSAS MagSANSFit")

# ----------------------------------------
# INIT
# ----------------------------------------

st.set_page_config(layout="wide")

initialize_state()
initialize_variable_values()
apply_pending_preset()
# ----------------------------------------
# DATA LOAD, Fit Options
# ----------------------------------------
treated_input = render_data_loader()

analysis_mode = st.session_state.global_model_state["analysis_mode"]
dataset_name = "Nuclear Signal"
st.session_state.active_dataset = (
        dataset_name
    )
if analysis_mode == "Nuclear-Magnetic Joint Analysis":

    # ----------------------------------------
    # DATASET SWITCHER
    # ----------------------------------------
    dataset_name = st.sidebar.radio(

        "Select Dataset",

        [
            "Nuclear Signal",
            "Magnetic Signal"
        ]
    )

    st.session_state.active_dataset = (
        dataset_name
    )

dataset = st.session_state.datasets[
    dataset_name
]

# ----------------------------------------
# MODEL SELECTION
# ----------------------------------------

render_model_selector()

# ----------------------------------------
# VARIABLE VISIBILITY
# ----------------------------------------

update_variable_activity(dataset)

# ----------------------------------------
# CONTROLS
# ----------------------------------------

render_controls(

    dataset
)

# ----------------------------------------
# PRESETS
# ----------------------------------------

st.sidebar.header("Advanced Setting Presets")

preset_name = st.sidebar.text_input(
    "Preset Name"
)

if st.sidebar.button("Save Preset"):

    save_preset(preset_name)

preset_files = [

    f.replace(".json", "")

    for f in os.listdir("presets")
]

if preset_files:

    selected_preset = st.sidebar.selectbox(

        "Load Preset",

        preset_files
    )

    if st.sidebar.button("Load Preset"):

        load_preset(selected_preset)

# ----------------------------------------
# ADVANCED SETTINGS
# ----------------------------------------

if st.sidebar.button("Open Advanced Settings"):

    render_settings_editor()

    validate_all_variable_values()

if st.sidebar.button("Analysis Configurations"):
    render_misc_controls()

# ----------------------------------------
# PLOT
# ----------------------------------------
if treated_input:
    current_variables = get_variable_config(st.session_state.datasets[st.session_state.active_dataset]["values"],st.session_state.global_model_state)
    current_model = get_model_config(st.session_state.global_model_state)
    fig,chi2 = single_plot(treated_input,st.session_state.global_model_state,st.session_state.active_dataset,current_variables,st.session_state.global_model_state["magnetic_scattering_background"])
    #fig,chi2_nuc,chi2_mag = create_plot(treated_input)

    st.plotly_chart(

        fig,

        width="stretch"
    )
    
    BO_config = get_BO_config(st.session_state.settings_df,st.session_state.datasets["Nuclear Signal"]["values"],st.session_state.datasets["Magnetic Signal"]["values"],st.session_state.global_model_state)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("chi2_red:", chi2)

    with col2:
        if "single_fit_result" not in st.session_state:
            st.session_state.single_fit_result = {}
        if st.button("Single Analysis"):
            st.write("Started Single Dataset Analysis")
            fit_result = singular_fit(treated_input,st.session_state.global_model_state,st.session_state.analysis_configs,st.session_state.active_dataset,current_variables,BO_config)
            st.session_state.single_fit_result[st.session_state.active_dataset]=fit_result
            st.write("Analysis Finished")

    with col3:
        if st.session_state.global_model_state["analysis_mode"] == "Nuclear-Magnetic Joint Analysis":
            if "joint_fit_result" not in st.session_state:
                st.session_state.joint_fit_result = {}
            if st.button("Joint Analysis"):
                st.write("Started Joint Dataset Analysis")
                if st.session_state.single_fit_result.get("Nuclear Signal") is None:
                    st.session_state.single_fit_result["Nuclear Signal"] = singular_fit(treated_input,st.session_state.global_model_state,st.session_state.analysis_configs,"Nuclear Signal",current_variables,BO_config,output_graph_data=False)
                
                if st.session_state.single_fit_result.get("Magnetic Signal") is None:
                    st.session_state.single_fit_result["Magnetic Signal"] = singular_fit(treated_input,st.session_state.global_model_state,st.session_state.analysis_configs,"Magnetic Signal",current_variables,BO_config,output_graph_data=False)
                joint_result = joint_analysis(treated_input,st.session_state.global_model_state,st.session_state.analysis_configs,st.session_state.single_fit_result,BO_config)
                st.session_state.joint_fit_result[st.session_state.active_dataset]=joint_result
                st.write("Analysis Finished")
        
    #    st.write("chi2_nuc:", chi2_nuc)
#
    #with col2:
    #    if chi2_mag:
    #        st.write("chi2_mag:", chi2_mag)
    
# ----------------------------------------
# DEBUG
# ----------------------------------------


#    with st.expander("Debug"):
#        st.write(BO_config)
#        st.write(st.session_state["data_loader"].name)
#        st.write(st.session_state.single_fit_result[st.session_state.active_dataset])