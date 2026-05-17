import json
import os

import pandas as pd
import streamlit as st


PRESET_DIR = "presets"

os.makedirs(PRESET_DIR, exist_ok=True)


def save_preset(name):

    data = {

        "settings_df": (
            st.session_state.settings_df
            .to_dict()
        )
    }

    filepath = os.path.join(
        PRESET_DIR,
        f"{name}.json"
    )

    with open(filepath, "w") as f:

        json.dump(data, f, indent=4)


def load_preset(name):

    filepath = os.path.join(
        PRESET_DIR,
        f"{name}.json"
    )

    with open(filepath, "r") as f:

        data = json.load(f)
    # -----------------------------------
    # STORE TEMPORARILY
    # -----------------------------------

    st.session_state.pending_preset = data

    # -----------------------------------
    # FORCE RERUN
    # -----------------------------------

    st.rerun()
def apply_pending_preset():

    if "pending_preset" not in st.session_state:
        return

    data = st.session_state.pending_preset

    # -----------------------------------
    # RESTORE SETTINGS
    # -----------------------------------

    st.session_state.settings_df = (
        pd.DataFrame(
            data["settings_df"]
        )
    )

    # -----------------------------------
    # CLEANUP
    # -----------------------------------

    del st.session_state.pending_preset