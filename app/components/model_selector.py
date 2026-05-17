import streamlit as st
"""
render_model_selector: Displays options for model selection
"""

def render_model_selector():

    if "distribution_type" not in st.session_state:

        st.session_state.distribution_type = (

            st.session_state
            .global_model_state["distribution_type"]
        )

    distribution_type = st.sidebar.selectbox(

        "Distribution Type",

        [
            "Single",
            "Double"
        ],

        key="distribution_type"
    )

    st.session_state.global_model_state["distribution_type"] = distribution_type

    if "distribution_1" not in st.session_state:

        st.session_state.distribution_1 = (

            st.session_state
            .global_model_state["distribution_1"]
        )

    distribution_1 = st.sidebar.selectbox(

        "Distribution 1",

        [
            "mono",
            "normal",
            "log_normal",
        ],

        key="distribution_1"
    )

    st.session_state.global_model_state["distribution_1"] = distribution_1

    if "model_1" not in st.session_state:

        st.session_state.model_1 = (

            st.session_state
            .global_model_state["model_1"]
        )

    model_1 = st.sidebar.selectbox(

        "Model 1",

        [
            "sphere",
            "ellipsoid",
            "core-shell",
        ],

        key="model_1"
    )

    st.session_state.global_model_state["model_1"] = model_1

    if distribution_type == "Double": #only render 2nd distribution if model is double
        if "distribution_2" not in st.session_state:

            st.session_state.distribution_2 = (

                st.session_state
                .global_model_state["distribution_2"]
            )

        distribution_2 = st.sidebar.selectbox(

            "Distribution 2",

            [
                "mono",
                "normal",
                "log_normal",
            ],

            key="distribution_2"
        )

        st.session_state.global_model_state["distribution_2"] = distribution_2

        if "model_2" not in st.session_state:

            st.session_state.model_2 = (

                st.session_state
                .global_model_state["model_2"]
            )

        model_2 = st.sidebar.selectbox(

            "Model 2",

            [
                "sphere",
                "ellipsoid",
                "core-shell",
            ],

            key="model_2"
        )

        st.session_state.global_model_state["model_2"] = model_2

    
