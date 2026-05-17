import streamlit as st
from core.utils.file_reader import file_reader_1d,file_reader_1d_nofilter,file_reader_2d,file_reader_2d_noise_filter
def render_data_loader():
    if "analysis_mode" not in st.session_state:

        st.session_state.analysis_mode = (

            st.session_state
            .global_model_state["analysis_mode"]
        )

    analysis_mode = st.sidebar.selectbox(

        "Analysis Mode",

        [
            "Distribution Fitting",
            "Nuclear-Magnetic Joint Analysis"
        ],

        key="analysis_mode"
    )
    st.session_state.global_model_state["analysis_mode"] = analysis_mode

    if analysis_mode == "Nuclear-Magnetic Joint Analysis":
        if "magnetic_scattering_background" not in st.session_state:

            st.session_state.magnetic_scattering_background = (

                st.session_state
                .global_model_state["magnetic_scattering_background"]
            )

        magnetic_scattering_background = st.sidebar.selectbox(

            "Magnetic Scattering Background",

            [
                "On",
                "Off"
            ],

            key="magnetic_scattering_background"
        )
        st.session_state.global_model_state["magnetic_scattering_background"] = magnetic_scattering_background
    uploaded_file = st.sidebar.file_uploader(
        "Load Data File",
        type=["csv", "txt"],
        key="data_loader"
    )
    if uploaded_file:
        if analysis_mode == "Distribution Fitting":
            Q,I,sigmaI,sigmaQ = file_reader_1d(uploaded_file)
            return Q,I,sigmaI,sigmaQ
        elif analysis_mode == "Nuclear-Magnetic Joint Analysis":
            Q,I_nuc,sigmaI_nuc,sigmaQ_nuc,I_mag,sigmaI_mag = file_reader_2d_noise_filter(uploaded_file)
            return Q,I_nuc,sigmaI_nuc,sigmaQ_nuc,I_mag,sigmaI_mag
        
    