import streamlit as st
from app.prototypes.variable import Variable
from functools import partial
"""
render_controls: Displays variable slidebar, input, and communicates value
"""
def render_controls(dataset):

    #settings_df = st.session_state.settings_df

    #values = dataset["values"]

    st.sidebar.header("Parameters")

    for variable in dataset["variables"].values():

        if not variable.active:
            continue
        
        

        min_value = float(
            variable.get_setting("min")
        )

        max_value = float(
            variable.get_setting("max")
        )

        default = float(
            variable.get_setting("default")
        )

        step = float(
            variable.get_setting("step")
        )

        current_value = (
            variable.get_value()
        )

        widget_key = (
            variable.widget_key
        )

        # -----------------------------------
        # LABEL
        # -----------------------------------

        st.sidebar.markdown(
            f"### {variable.display_name}"
        )

        # -----------------------------------
        # Initialize common key
        # -----------------------------------

        #shared_key = f"{widget_key}_value"
        slider_key = f"{widget_key}_slider"
        input_key = f"{widget_key}_input"
        #initialize keys to avoid problems during switch
        if slider_key not in st.session_state:

            st.session_state[slider_key] = (
                float(variable.get_value())
            )
        if input_key not in st.session_state:

            st.session_state[input_key] = (
                float(variable.get_value())
            )

        #def on_slider_change():
        #    st.session_state[shared_key] = st.session_state[f"{widget_key}_slider"]
#
#
        #def on_number_change():
        #    st.session_state[shared_key] = st.session_state[f"{widget_key}_input"]
        # -----------------------------------
        # SLIDER
        # -----------------------------------

        #update logic
        def update_slider(s_key=slider_key, i_key=input_key):
            st.session_state[s_key] = st.session_state[i_key]

        def update_numin(s_key=slider_key, i_key=input_key):
            st.session_state[i_key] = st.session_state[s_key]

        #init_val = float(variable.get_value())
        slider_value = st.sidebar.slider(

            variable.display_name,

            min_value=min_value,

            max_value=max_value,

            value=float(st.session_state[slider_key]),

            step=step,

            key=slider_key,
            on_change = update_numin,
            label_visibility="collapsed"
        )

        #st.session_state[shared_key] = slider_value


        # -----------------------------------
        # MANUAL INPUT
        # -----------------------------------

        input_value = st.sidebar.number_input(

            "Value",

            min_value=min_value,

            max_value=max_value,

            value=float(st.session_state[input_key]),

            step=step,

            key=input_key,
            on_change=update_slider
        )
        
        variable.set_value(input_value)
