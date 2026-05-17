import sys
import os

sys.path.append(os.getcwd())

import streamlit as st

from app.config.variables_lib import VARIABLES
from components.controls import render_variable

st.set_page_config(layout="wide")

st.title("Curve Visualizer")

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header("Controls")

values = {}

for variable in VARIABLES:

    values[variable.name] = render_variable(variable)

# ------------------------------------------------
# DEBUG OUTPUT
# ------------------------------------------------

st.write(values)