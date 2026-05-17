import streamlit as st
def validate_all_variable_values():

    for dataset in (
        st.session_state.datasets.values()
    ):

        for variable in (
            dataset["variables"].values()
        ):

            current = variable.get_value()

            variable.set_value(current)