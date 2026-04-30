import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
#from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize
from core.models.assembled_problem import nano_intensity_spheroid,final_intensity_spheroid
from problems.coreshell import formfactor_coreshell
from problems.distributions import distribution_normal,distribution_lognormal
from problems.ellipsoid import volume_ellipsoid,formfactor_ellipsoid
from problems.sphere import volume_sphere,formfactor_sphere
from problems.approximations import I_porod
from core.models.loss_functions import loss_chi2
from core.utils.helper_functions import reward_variance
import warnings

torch.set_default_dtype(torch.double)
torch.manual_seed(0)

"""
BO model with fixed entries, used when one fit result (nuc/mag) needs to be fixed.
"""

def chi2_final_intensity_spheroid_BO_single_fixed(theta,distribution,formfactor,volume,list_q,experiment,uncertainty,rm,sigma=None,k=None):
    theta_np = theta.detach().cpu().numpy()
    num_evaluations= list_q.size
    results_chi2 = []
    #check model for each distribution
    if callable(distribution):
        if formfactor is formfactor_sphere:
            num_parameters = 5
        elif formfactor is formfactor_ellipsoid:
            num_parameters = 6
        elif formfactor is formfactor_coreshell:
            num_parameters = 7
    else:
        if formfactor is formfactor_sphere:
            num_parameters = 4
        elif formfactor is formfactor_ellipsoid:
            num_parameters = 5
        elif formfactor is formfactor_coreshell:
            num_parameters = 6
    # logic for different cases
    for t in theta_np:
        Ibg = np.exp(t[0])
        C = np.exp(t[1])
        combinedfactor = np.exp(-t[2])
        sigma_rm = None
        k_theta = None
        mu_theta = None
        model_prediction = np.zeros(num_evaluations)
        if callable(distribution):
            if num_parameters == 5:
                sigma_rm = rm*sigma
            elif num_parameters == 6:
                sigma_rm = rm*sigma
                k_theta = k
            elif num_parameters == 7:
                sigma_rm = rm*sigma
                k_theta = k
                mu_theta = t[3]
        else:
            if num_parameters == 5:
                k_theta = k
            elif num_parameters == 6:
                k_theta = k
                mu_theta = t[3]
        for i in range(num_evaluations):
            contribution_1 = final_intensity_spheroid(I_porod,combinedfactor,distribution,formfactor,volume,Ibg,list_q[i],rm,C,sigma=sigma_rm,k=k_theta,mu=mu_theta)
            model_prediction[i] = contribution_1
        chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        results_chi2.append(chi2)

    return torch.tensor(results_chi2, dtype=torch.double)
"""
when bg and porod is removed, use the following nano intensity variant
"""
def chi2_nano_intensity_spheroid_BO_single_fixed(theta,distribution,formfactor,volume,list_q,experiment,uncertainty,rm,sigma=None,k=None):
    theta_np = theta.detach().cpu().numpy()
    num_evaluations= list_q.size
    results_chi2 = []
    #check model for each distribution
    if callable(distribution):
        if formfactor is formfactor_sphere:
            num_parameters = 3
        elif formfactor is formfactor_ellipsoid:
            num_parameters = 4
        elif formfactor is formfactor_coreshell:
            num_parameters = 5
    else:
        if formfactor is formfactor_sphere:
            num_parameters = 2
        elif formfactor is formfactor_ellipsoid:
            num_parameters = 3
        elif formfactor is formfactor_coreshell:
            num_parameters = 4
    # logic for different cases
    for t in theta_np:
        combinedfactor = np.exp(-t[0])
        sigma_rm = None
        k_theta = None
        mu_theta = None
        model_prediction = np.zeros(num_evaluations)
        if callable(distribution):
            if num_parameters == 3:
                sigma_rm = rm*sigma
            elif num_parameters == 4:
                sigma_rm = rm*sigma
                k_theta = k
            elif num_parameters == 5:
                sigma_rm = rm*sigma
                k_theta = k
                mu_theta = t[1]
        else:
            if num_parameters == 3:
                k_theta = k
            elif num_parameters == 4:
                k_theta = k
                mu_theta = t[1]
        for i in range(num_evaluations):
            contribution_1 = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,list_q[i],rm,sigma=sigma_rm,k=k_theta,mu=mu_theta)
            model_prediction[i] = contribution_1
        chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        results_chi2.append(chi2)

    return torch.tensor(results_chi2, dtype=torch.double)

def chi2_final_intensity_spheroid_BO_double_fixed(theta, distribution_1, distribution_2, formfactor_1, formfactor_2 ,volume_1, volume_2,list_q,experiment,uncertainty,rm_1,rm_2,sigma_1=None,k_1=None, sigma_2=None,k_2=None):
    theta_np = theta.detach().cpu().numpy()
    #num_parameters = theta_np.shape[1] #record number of columns
    num_evaluations= list_q.size
    results_chi2 = []
    #check model for each distribution
    
    # logic for different cases
    for t in theta_np:
        sigma_rm_1 = None
        sigma_rm_2 = None
        k_theta_1 = None
        k_theta_2 = None
        mu_theta_1 = None
        mu_theta_2 = None
        if callable(distribution_1):
            if formfactor_1 is formfactor_sphere:
                num_parameters_1 = 3
                true_num_parameters_1 = 5
                sigma_rm_1 = rm_1*sigma_1
            elif formfactor_1 is formfactor_ellipsoid:
                num_parameters_1 = 3
                true_num_parameters_1 = 6
                sigma_rm_1 = rm_1*sigma_1
                k_theta_1 = k_1
            elif formfactor_1 is formfactor_coreshell:
                num_parameters_1 = 4
                true_num_parameters_1 = 7
                sigma_rm_1 = rm_1*sigma_1
                k_theta_1 = k_1
                mu_theta_1 = t[3]
        else:
            if formfactor_1 is formfactor_sphere:
                num_parameters_1 = 3
                true_num_parameters_1 = 4
                
            elif formfactor_1 is formfactor_ellipsoid:
                num_parameters_1 = 3
                true_num_parameters_1 = 5
                k_theta_1 = k_1
            elif formfactor_1 is formfactor_coreshell:
                true_num_parameters_1 = 6
                num_parameters_1 = 4
                k_theta_1 = k_1
                mu_theta_1 = t[3]
        if callable(distribution_2):
            if formfactor_2 is formfactor_sphere:
                num_parameters_2 = 1
                true_num_parameters_2 = 3
                sigma_rm_2 = rm_2*sigma_2
            elif formfactor_2 is formfactor_ellipsoid:
                num_parameters_2 = 1
                true_num_parameters_2 = 4
                sigma_rm_2 = rm_2*sigma_2
                k_theta_2 = k_2
            elif formfactor_2 is formfactor_coreshell:
                num_parameters_2 = 2
                true_num_parameters_2 = 5
                sigma_rm_2 = rm_2*sigma_2
                k_theta_2 = k_2
                mu_theta_2 = t[num_parameters_1+1]
        else:
            if formfactor_2 is formfactor_sphere:
                num_parameters_2 = 1
                true_num_parameters_2 = 2
            elif formfactor_2 is formfactor_ellipsoid:
                num_parameters_2 = 1
                true_num_parameters_2 = 3
                k_theta_2 = k_2
            elif formfactor_2 is formfactor_coreshell:
                num_parameters_2 = 2
                true_num_parameters_2 = 4
                k_theta_2 = k_2
                mu_theta_2 = t[num_parameters_1+1]
        Ibg = np.exp(t[0])
        C = np.exp(t[1])
        combinedfactor_1 = np.exp(-t[2])
        combinedfactor_2 = np.exp(-t[num_parameters_1])
        model_prediction = np.zeros(num_evaluations)
        for i in range(num_evaluations):
            contribution_1 = final_intensity_spheroid(I_porod,combinedfactor_1,distribution_1,formfactor_1,volume_1,Ibg,list_q[i],rm_1,C,sigma=sigma_rm_1,k=k_theta_1,mu=mu_theta_1)
            contribution_2 = nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,list_q[i],rm_2,sigma=sigma_rm_2,k=k_theta_2,mu=mu_theta_2)
            model_prediction[i] = contribution_1+contribution_2
        num_parameters = true_num_parameters_1+true_num_parameters_2
        chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        results_chi2.append(chi2)

    return torch.tensor(results_chi2, dtype=torch.double)

def chi2_nano_intensity_spheroid_BO_double_fixed(theta, distribution_1, distribution_2, formfactor_1, formfactor_2 ,volume_1, volume_2,list_q,experiment,uncertainty,rm_1,rm_2,sigma_1=None,k_1=None, sigma_2=None,k_2=None):
    theta_np = theta.detach().cpu().numpy()
    #num_parameters = theta_np.shape[1] #record number of columns
    num_evaluations= list_q.size
    results_chi2 = []
    #check model for each distribution
    
    # logic for different cases
    for t in theta_np:
        sigma_rm_1 = None
        sigma_rm_2 = None
        k_theta_1 = None
        k_theta_2 = None
        mu_theta_1 = None
        mu_theta_2 = None
        if callable(distribution_1):
            if formfactor_1 is formfactor_sphere:
                num_parameters_1 = 1
                true_num_parameters_1 = 3
                sigma_rm_1 = rm_1*sigma_1
            elif formfactor_1 is formfactor_ellipsoid:
                num_parameters_1 = 1
                true_num_parameters_1 = 4
                sigma_rm_1 = rm_1*sigma_1
                k_theta_1 = k_1
            elif formfactor_1 is formfactor_coreshell:
                num_parameters_1 = 2
                true_num_parameters_1 = 5
                sigma_rm_1 = rm_1*sigma_1
                k_theta_1 = k_1
                mu_theta_1 = t[1]
        else:
            if formfactor_1 is formfactor_sphere:
                num_parameters_1 = 1
                true_num_parameters_1 = 2
                
            elif formfactor_1 is formfactor_ellipsoid:
                num_parameters_1 = 1
                true_num_parameters_1 = 3
                k_theta_1 = k_1
            elif formfactor_1 is formfactor_coreshell:
                true_num_parameters_1 = 4
                num_parameters_1 = 2
                k_theta_1 = k_1
                mu_theta_1 = t[1]
        if callable(distribution_2):
            if formfactor_2 is formfactor_sphere:
                num_parameters_2 = 1
                true_num_parameters_2 = 3
                sigma_rm_2 = rm_2*sigma_2
            elif formfactor_2 is formfactor_ellipsoid:
                num_parameters_2 = 1
                true_num_parameters_2 = 4
                sigma_rm_2 = rm_2*sigma_2
                k_theta_2 = k_2
            elif formfactor_2 is formfactor_coreshell:
                num_parameters_2 = 2
                true_num_parameters_2 = 5
                sigma_rm_2 = rm_2*sigma_2
                k_theta_2 = k_2
                mu_theta_2 = t[num_parameters_1+1]
        else:
            if formfactor_2 is formfactor_sphere:
                num_parameters_2 = 1
                true_num_parameters_2 = 2
            elif formfactor_2 is formfactor_ellipsoid:
                num_parameters_2 = 1
                true_num_parameters_2 = 3
                k_theta_2 = k_2
            elif formfactor_2 is formfactor_coreshell:
                num_parameters_2 = 2
                true_num_parameters_2 = 4
                k_theta_2 = k_2
                mu_theta_2 = t[num_parameters_1+1]
        combinedfactor_1 = np.exp(-t[0])
        combinedfactor_2 = np.exp(-t[num_parameters_1])
        model_prediction = np.zeros(num_evaluations)
        for i in range(num_evaluations):
            contribution_1 = nano_intensity_spheroid(combinedfactor_1,distribution_1,formfactor_1,volume_1,list_q[i],rm_1,sigma=sigma_rm_1,k=k_theta_1,mu=mu_theta_1)
            contribution_2 = nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,list_q[i],rm_2,sigma=sigma_rm_2,k=k_theta_2,mu=mu_theta_2)
            model_prediction[i] = contribution_1+contribution_2
        num_parameters = true_num_parameters_1+true_num_parameters_2
        chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        results_chi2.append(chi2)

    return torch.tensor(results_chi2, dtype=torch.double)


