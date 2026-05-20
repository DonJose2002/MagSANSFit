import streamlit as st

@st.dialog("Advanced Settings")
def render_settings_editor():

    # -----------------------------------
    # INTERNAL -> DISPLAY
    # -----------------------------------

    display_df = (
        st.session_state.settings_df
        .transpose()
        .reset_index()
    )

    display_df.rename(

        columns={
            "index": "variable"
        },

        inplace=True
    )

    # -----------------------------------
    # EDITOR
    # -----------------------------------

    edited_display_df = st.data_editor(

        display_df,

        width='stretch',

        num_rows="dynamic"
    )

    # -----------------------------------
    # DISPLAY -> INTERNAL
    # -----------------------------------

    updated_internal_df = (

        edited_display_df
        .set_index("variable")
        .transpose()
    )

    st.session_state.settings_df = (
        updated_internal_df
    )