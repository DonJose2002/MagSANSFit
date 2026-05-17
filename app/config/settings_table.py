import pandas as pd
#settings_df = pd.DataFrame({
#    "Settings": ["min", "max", "searchwidth", "searchwidth_jointfit","searchwidth_fine","step"],
#    "log_Ibg":    [ -10,   1,   3,   1, 0.5, 0.01],
#    "log_C":      [ -10,   1,   3,   1, 0.5, 0.01],
#    "A":          [ 0.1,  10,   2,   1, 0.5, 0.01],
#    "Rm":         [ 0.1,  20,   4,   1, 0.5, 0.01],
#    "Sigma":      [0.01, 0.3, 0.1, 0.1, 0.1, 0.01],
#    "kellipsoid": [ 0.5,   3, 0.5, 0.5, 0.5, 0.01],
#    "kshell":     [   0,   1, 0.3, 0.3, 0.3, 0.01],
#    "mu":         [ -10,  10,   4,   1, 0.5, 0.01]
#})
"""
create_variable_settings: initializes advanced settings for variables
"""
def create_variable_settings():

    df = pd.DataFrame({

        "log_Ibg": {

            "min": -10.0,
            "max": 1.0,
            "default":-5.0,
            "searchwidth": 3.0,
            "searchwidth_jointfit": 1.0,
            "searchwidth_fine": 0.5,
            "step": 0.01
        },

        "log_C": {

            "min": -10.0,
            "max": 1.0,
            "default":-5.0,
            "searchwidth": 3.0,
            "searchwidth_jointfit": 1.0,
            "searchwidth_fine": 0.5,
            "step": 0.01
        },

        "A": {

            "min": 0.1,
            "max": 10.0,
            "default":5.0,
            "searchwidth": 2.0,
            "searchwidth_jointfit": 1.0,
            "searchwidth_fine": 0.5,
            "step": 0.01
        },

        "Rm": {

            "min": 0.1,
            "max": 20.0,
            "default":5.0,
            "searchwidth": 4.0,
            "searchwidth_jointfit": 1.0,
            "searchwidth_fine": 0.5,
            "step": 0.01
        },
        "Sigma": {

            "min": 0.01,
            "max": 0.3,
            "default":0.2,
            "searchwidth": 0.1,
            "searchwidth_jointfit": 0.1,
            "searchwidth_fine": 0.1,
            "step": 0.01
        },
        "kellipsoid": {

            "min": 0.5,
            "max": 3.0,
            "default":2.0,
            "searchwidth": 0.5,
            "searchwidth_jointfit": 0.5,
            "searchwidth_fine": 0.5,
            "step": 0.01
        },
        "kshell": {

            "min": 0.0,
            "max": 1.0,
            "default":0.5,
            "searchwidth": 0.3,
            "searchwidth_jointfit": 0.3,
            "searchwidth_fine": 0.3,
            "step": 0.01
        },
        "mu": {

            "min": -10.0,
            "max": 10.0,
            "default":5.0,
            "searchwidth": 4.0,
            "searchwidth_jointfit": 1.0,
            "searchwidth_fine": 0.5,
            "step": 0.01
        },
    })

    return df