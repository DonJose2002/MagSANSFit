import streamlit as st
"""
Variable class: defines operations for UI and UI-core usage
"""
class Variable:

    def __init__(

        self,

        internal_name,
        display_name,
        type_name,
        dataset_name
    ):

        self.internal_name = internal_name #internal identifier

        self.display_name = display_name #name showing in UI

        self.type_name = type_name #reference in property

        self.dataset_name = dataset_name #the dataset that Variable belongs to

        self.active = False
    # -----------------------------------
    # SETTINGS ACCESS
    # -----------------------------------

    def get_setting(self, property_name):

        settings_df = (
            st.session_state.settings_df
        )

        return settings_df.loc[

            property_name,

            self.type_name #variables with the same type_name share the same advanced settings
        ]

    # -----------------------------------
    # VALUE ACCESS
    # -----------------------------------

    def get_value(self):

        dataset = (
            st.session_state.datasets[
                self.dataset_name
            ]
        )

        return dataset["values"].get(

            self.internal_name,

            self.get_setting("default")
        )

    def set_value(self, value):

        dataset = (
            st.session_state.datasets[
                self.dataset_name
            ]
        )

        value = self.clamp_value(value)

        dataset["values"][
            self.internal_name
        ] = value

    def clamp_value(self, value):

        min_value = self.get_setting("min")

        max_value = self.get_setting("max")
        

        return max(min_value,min(max_value,value))
    # -----------------------------------
    # STREAMLIT WIDGET KEY
    # -----------------------------------

    @property
    def widget_key(self):

        return (

            f"{self.dataset_name}_"
            f"{self.internal_name}"
        )