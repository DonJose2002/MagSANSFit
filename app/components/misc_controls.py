import streamlit as st

@st.dialog("Analysis Options")
def render_misc_controls():
    for entry in st.session_state.analysis_configs.items():
        name, init_value = entry
        current_key = f"{name}_input"
        if current_key not in st.session_state:
            st.session_state[current_key] = init_value
        if isinstance(init_value,float):
            step = 0.01
            format = "%.6f"
            min_val = 0.0
        else:
            step = 1
            format = None
            min_val = 0
        input_value = st.number_input(
            name,
            min_value = min_val,
            step=step,
            format=format,
            key=current_key,
        )
        st.session_state.analysis_configs[name] = input_value
