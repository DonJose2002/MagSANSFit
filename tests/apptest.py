import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Curve Visualizer",
    layout="wide"
)

st.title("Curve Visualizer")

# ---------------------------------------------------
# SESSION STATE DEFAULTS
# ---------------------------------------------------

if "slider_min" not in st.session_state:
    st.session_state.slider_min = 0.0

if "slider_max" not in st.session_state:
    st.session_state.slider_max = 10.0

if "curve_color" not in st.session_state:
    st.session_state.curve_color = "blue"

# ---------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------

st.sidebar.header("Controls")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Load Data File",
    type=["csv", "txt"]
)

# Dropdown menu
curve_type = st.sidebar.selectbox(
    "Curve Type",
    ["Sine", "Cosine", "Quadratic"]
)

# Slider
amplitude = st.sidebar.slider(
    "Amplitude",
    min_value=st.session_state.slider_min,
    max_value=st.session_state.slider_max,
    value=1.0
)

# Manual input
frequency = st.sidebar.number_input(
    "Frequency",
    value=1.0,
    step=0.1
)

# String input
plot_title = st.sidebar.text_input(
    "Plot Title",
    value="My Curve"
)

@st.dialog("Advanced Settings")
def settings_dialog():

    st.write("Adjust advanced settings")

    st.session_state.slider_min = st.number_input(
        "Slider Minimum",
        value=st.session_state.slider_min
    )

    st.session_state.slider_max = st.number_input(
        "Slider Maximum",
        value=st.session_state.slider_max
    )

    st.session_state.curve_color = st.selectbox(
        "Curve Color",
        ["blue", "red", "green", "orange"]
    )

    st.write("Settings saved automatically.")

if st.sidebar.button("Open Settings"):
    settings_dialog()
# ---------------------------------------------------
# SETTINGS DIALOG
# ---------------------------------------------------
# expander
#with st.sidebar.expander("Advanced Settings"):
#
#    st.session_state.slider_min = st.number_input(
#        "Slider Minimum",
#        value=st.session_state.slider_min
#    )
#
#    st.session_state.slider_max = st.number_input(
#        "Slider Maximum",
#        value=st.session_state.slider_max
#    )
#
#    st.session_state.curve_color = st.selectbox(
#        "Curve Color",
#        ["blue", "red", "green", "orange"]
#    )
# popover
#with st.sidebar.popover("Settings"):
#
#    st.write("Advanced Settings")
#
#    st.session_state.slider_min = st.number_input(
#        "Slider Minimum",
#        value=st.session_state.slider_min
#    )
#
#    st.session_state.slider_max = st.number_input(
#        "Slider Maximum",
#        value=st.session_state.slider_max
#    )
#
#    st.session_state.curve_color = st.selectbox(
#        "Curve Color",
#        ["blue", "red", "green", "orange"]
#    )
# DATA LOADING
# ---------------------------------------------------

data = None

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully")

    st.subheader("Preview")
    st.dataframe(data.head())

# ---------------------------------------------------
# CURVE GENERATION
# ---------------------------------------------------

#x = np.linspace(0, 10, 1000)
#
#if curve_type == "Sine":
#    y = amplitude * np.sin(frequency * x)
#
#elif curve_type == "Cosine":
#    y = amplitude * np.cos(frequency * x)
#
#elif curve_type == "Quadratic":
#    y = amplitude * (x ** 2)
#
## ---------------------------------------------------
## PLOTLY FIGURE
## ---------------------------------------------------
#
#fig = go.Figure()
#
#fig.add_trace(
#    go.Scatter(
#        x=x,
#        y=y,
#        mode="lines",
#        line=dict(color=st.session_state.curve_color),
#        name=curve_type
#    )
#)
#
#fig.update_layout(
#    title=plot_title,
#    xaxis_title="X",
#    yaxis_title="Y",
#    template="plotly_white"
#)
#
#st.plotly_chart(fig, width='stretch')