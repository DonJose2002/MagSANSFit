import sys
import os

sys.path.append(os.getcwd())

import streamlit as st

from app.utilities.state import initialize_state,initialize_variable_values
from app.utilities.visibility import update_variable_activity
from app.utilities.validation import validate_all_variable_values
from app.components.controls import render_controls
from app.components.model_selector import render_model_selector
from app.components.plotting import create_plot
from app.components.settings_editor import render_settings_editor
from app.components.preset_manager import save_preset,load_preset
from app.components.preset_manager import apply_pending_preset
from app.components.data_loader import render_data_loader
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

st.sidebar.header("Presets")

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

# ----------------------------------------
# PLOT
# ----------------------------------------
if treated_input:
    fig,chi2_nuc,chi2_mag = create_plot(treated_input)

    st.plotly_chart(

        fig,

        width="stretch"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("chi2_nuc:", chi2_nuc)

    with col2:
        if chi2_mag:
            st.write("chi2_mag:", chi2_mag)

# ----------------------------------------
# DEBUG
# ----------------------------------------

with st.expander("Debug"):
    for variable in dataset["variables"].values():
        st.write(dataset_name)
        st.write(variable.dataset_name)
        st.write(variable.internal_name)
        st.write(variable.display_name)
        st.write(variable.get_value())