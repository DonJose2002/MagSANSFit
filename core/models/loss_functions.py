import numpy as np
import torch
from core.models.assembled_problem import final_intensity_spheroid
"""
Loss_chi2: Function for calculating chi squared loss.
experiment: input numpy array of experimental measurements at different Q
uncertainty: associated uncertainty
model: associated predicted values
num_parameters: number of parameters 
"""
def loss_chi2 (experiment, uncertainty, model, num_parameters):
    num_samples = np.size(experiment)
    return 1/(num_samples-num_parameters)*np.sum(((experiment-model)/uncertainty)**2)
