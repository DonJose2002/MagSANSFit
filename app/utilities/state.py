import streamlit as st

from app.prototypes.variable import Variable

from app.config.settings_table import (
    create_variable_settings
)

from app.config.variables_lib import (
    ALL_VARIABLES
)

def create_all_variables(dataset_name):

    variables = {}

    for name_tuple in ALL_VARIABLES:
        
        type_name, display_name, internal_name = name_tuple

        variables[internal_name] = (

            Variable(

                internal_name,
                display_name,
                type_name,
                dataset_name
            )
        )

    return variables

def create_empty_dataset(dataset_name):

    return {

        "values": {},

        "variables": create_all_variables(dataset_name),

    }




def initialize_state():

    if "settings_df" not in st.session_state:

        st.session_state.settings_df = (
            create_variable_settings()
        )
    if "global_model_state" not in st.session_state:

        st.session_state.global_model_state = {
        "analysis_mode": "Nuclear-Magnetic Joint Analysis",
        "magnetic_scattering_background": "On",
        "distribution_type": "Single",
        "distribution_1": "mono",
        "model_1":"sphere",
        "distribution_2": "mono",
        "model_2":"sphere",

    }
    if "analysis_configs" not in st.session_state:
        st.session_state.analysis_configs = {
            "max_iteration_BO": 75,
            "max_iteration_BO_Pareto": 50,
            "logspace_tolerance": 0.1,
            "n_init": 100,
            "n_init_discrepancy": 100,
            "n_init_fine": 50,
            "min_model_discrepancy": 1.0e-4,
            "max_model_discrepancy": 1.0
        }
        
    if "datasets" not in st.session_state:
        
        st.session_state.datasets = {

            "Nuclear Signal": create_empty_dataset("Nuclear Signal"),

            "Magnetic Signal": create_empty_dataset("Magnetic Signal")
        }

    if "active_dataset" not in st.session_state:

        st.session_state.active_dataset = (
            "Nuclear Signal"
        )

def initialize_variable_values():
    for dataset in st.session_state.datasets.values():
        values = dataset["values"]
        for variable in dataset["variables"].values():
            if variable.internal_name not in values:
                values[variable.internal_name] = variable.get_value()