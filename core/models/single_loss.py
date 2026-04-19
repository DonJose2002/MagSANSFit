"""
Loss functions for non-torch usage
"""
import numpy as np
import torch

from core.models.loss_functions import loss_chi2
from core.models.assembled_problem import nano_intensity_spheroid,final_intensity_spheroid
from problems.coreshell import formfactor_coreshell
from problems.distributions import distribution_normal,distribution_lognormal
from problems.ellipsoid import volume_ellipsoid,formfactor_ellipsoid
from problems.sphere import volume_sphere,formfactor_sphere
from problems.approximations import I_porod
"""
theta: numpy array of parameters
distribution,formfactor,volume: models
list_q:list of points to evaluate
experiment: measurements
uncertainty: error bar
"""
def chi2_final_intensity_spheroid_single(theta,distribution,formfactor,volume,list_q,experiment,uncertainty):
    num_parameters = theta.size #record number of columns
    num_evaluations= list_q.size    
    if callable(distribution): # check if distribution is mono
        if num_parameters == 5: #sphere case
            Ibg = np.exp(theta[0])
            C = np.exp(theta[1])
            combinedfactor = np.exp(-theta[2])
            rm = theta[3]
            sigma_rm = theta[3]*theta[4]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = final_intensity_spheroid(I_porod,combinedfactor,distribution,formfactor,volume,Ibg,list_q[i],rm,C,sigma=sigma_rm)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        elif num_parameters == 6: #ellipsoid case
            Ibg = np.exp(theta[0])
            C = np.exp(theta[1])
            combinedfactor = np.exp(-theta[2])
            rm = theta[3]
            sigma_rm = theta[3]*theta[4]
            k_theta = theta[5]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = final_intensity_spheroid(I_porod,combinedfactor,distribution,formfactor,volume,Ibg,list_q[i],rm,C,sigma=sigma_rm,k=k_theta)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        elif num_parameters == 7: #core-shell case
            Ibg = np.exp(theta[0])
            C = np.exp(theta[1])
            combinedfactor = np.exp(-theta[2])
            rm = theta[3]
            sigma_rm = theta[3]*theta[4]
            k_theta = theta[5]
            mu_theta = theta[6]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = final_intensity_spheroid(I_porod,combinedfactor,distribution,formfactor,volume,Ibg,list_q[i],rm,C,sigma=sigma_rm,k=k_theta,mu=mu_theta)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
    else: #monodistribution
        if num_parameters == 4: #sphere case
            Ibg = np.exp(theta[0])
            C = np.exp(theta[1])
            combinedfactor = np.exp(-theta[2])
            rm = theta[3]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = final_intensity_spheroid(I_porod,combinedfactor,distribution,formfactor,volume,Ibg,list_q[i],rm,C)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        elif num_parameters == 5: #ellipsoid case
            Ibg = np.exp(theta[0])
            C = np.exp(theta[1])
            combinedfactor = np.exp(-theta[2])
            rm = theta[3]
            k_theta = theta[4]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = final_intensity_spheroid(I_porod,combinedfactor,distribution,formfactor,volume,Ibg,list_q[i],rm,C,k=k_theta)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        elif num_parameters == 6: #core-shell case
            Ibg = np.exp(theta[0])
            C = np.exp(theta[1])
            combinedfactor = np.exp(-theta[2])
            rm = theta[3]
            k_theta = theta[4]
            mu_theta = theta[5]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = final_intensity_spheroid(I_porod,combinedfactor,distribution,formfactor,volume,Ibg,list_q[i],rm,C,k=k_theta,mu=mu_theta)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
    return chi2
"""
when bg and porod is removed, use the following nano intensity variant
"""
def chi2_nano_intensity_spheroid_single(theta,distribution,formfactor,volume,list_q,experiment,uncertainty):
    num_parameters = theta.size #record number of columns
    num_evaluations= list_q.size    
    if callable(distribution): # check if distribution is mono
        if num_parameters == 3: #sphere case
            combinedfactor = np.exp(-theta[0])
            rm = theta[1]
            sigma_rm = theta[1]*theta[2]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,list_q[i],rm,sigma=sigma_rm)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        elif num_parameters == 4: #ellipsoid case
            combinedfactor = np.exp(-theta[0])
            rm = theta[1]
            sigma_rm = theta[1]*theta[2]
            k_theta = theta[3]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,list_q[i],rm,sigma=sigma_rm,k=k_theta)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        elif num_parameters == 5: #core-shell case
            combinedfactor = np.exp(-theta[0])
            rm = theta[1]
            sigma_rm = theta[1]*theta[2]
            k_theta = theta[3]
            mu_theta = theta[4]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,list_q[i],rm,sigma=sigma_rm,k=k_theta,mu=mu_theta)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
    else: #monodistribution
        if num_parameters == 2: #sphere case
            combinedfactor = np.exp(-theta[0])
            rm = theta[1]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,list_q[i],rm)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        elif num_parameters == 3: #ellipsoid case
            combinedfactor = np.exp(-theta[0])
            rm = theta[1]
            k_theta = theta[2]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,list_q[i],rm,k=k_theta)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
        elif num_parameters == 4: #core-shell case
            combinedfactor = np.exp(-theta[0])
            rm = theta[1]
            k_theta = theta[2]
            mu_theta = theta[3]
            model_prediction = np.zeros(num_evaluations)
            for i in range(num_evaluations): 
                model_prediction[i] = nano_intensity_spheroid(combinedfactor,distribution,formfactor,volume,list_q[i],rm,k=k_theta,mu=mu_theta)
            chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)
    return chi2

def chi2_final_intensity_spheroid_double(theta, distribution_1, distribution_2, formfactor_1, formfactor_2 ,volume_1, volume_2,list_q,experiment,uncertainty):
    num_parameters = theta.size #record number of columns
    num_evaluations= list_q.size
    #check model for each distribution
    if callable(distribution_1):
        if formfactor_1 is formfactor_sphere:
            num_parameters_1 = 5
        elif formfactor_1 is formfactor_ellipsoid:
            num_parameters_1 = 6
        elif formfactor_1 is formfactor_coreshell:
            num_parameters_1 = 7
    else:
        if formfactor_1 is formfactor_sphere:
            num_parameters_1 = 4
        elif formfactor_1 is formfactor_ellipsoid:
            num_parameters_1 = 5
        elif formfactor_1 is formfactor_coreshell:
            num_parameters_1 = 6
    if callable(distribution_2):
        if formfactor_2 is formfactor_sphere:
            num_parameters_2 = 3
        elif formfactor_2 is formfactor_ellipsoid:
            num_parameters_2 = 4
        elif formfactor_2 is formfactor_coreshell:
            num_parameters_2 = 5
    else:
        if formfactor_2 is formfactor_sphere:
            num_parameters_2 = 2
        elif formfactor_2 is formfactor_ellipsoid:
            num_parameters_2 = 3
        elif formfactor_2 is formfactor_coreshell:
            num_parameters_2 = 4
    # logic for different cases

    Ibg = np.exp(theta[0])
    C = np.exp(theta[1])
    combinedfactor_1 = np.exp(-theta[2])
    combinedfactor_2 = np.exp(-theta[num_parameters_1])
    sigma_rm_1 = None
    sigma_rm_2 = None
    k_theta_1 = None
    k_theta_2 = None
    mu_theta_1 = None
    mu_theta_2 = None
    model_prediction = np.zeros(num_evaluations)
    if callable(distribution_1):
        if num_parameters_1 == 5:
            rm_1 = theta[3]
            sigma_rm_1 = theta[3]*theta[4]
        elif num_parameters_1 == 6:
            rm_1 = theta[3]
            sigma_rm_1 = theta[3]*theta[4]
            k_theta_1 = theta[5]
        elif num_parameters_1 == 7:
            rm_1 = theta[3]
            sigma_rm_1 = theta[3]*theta[4]
            k_theta_1 = theta[5]
            mu_theta_1 = theta[6]
    else:
        if num_parameters_1 == 4:
            rm_1 = theta[3]
        elif num_parameters_1 == 5:
            rm_1 = theta[3]
            k_theta_1 = theta[4]
        elif num_parameters_1 == 6:
            rm_1 = theta[3]
            k_theta_1 = theta[4]
            mu_theta_1 = theta[5]
    if callable(distribution_2):
        if num_parameters_2 == 3:
            rm_2 = theta[num_parameters_1+1]
            sigma_rm_2 = theta[num_parameters_1+1]*theta[num_parameters_1+2]
        elif num_parameters_2 == 4:
            rm_2 = theta[num_parameters_1+1]
            sigma_rm_2 = theta[num_parameters_1+1]*theta[num_parameters_1+2]
            k_theta_2 = theta[num_parameters_1+3]
        elif num_parameters_2 == 5:
            rm_2 = theta[num_parameters_1+1]
            sigma_rm_2 = theta[num_parameters_1+1]*theta[num_parameters_1+2]
            k_theta_2 = theta[num_parameters_1+3]
            mu_theta_2 = theta[num_parameters_1+4]
    else:
        if num_parameters_2 == 2:
            rm_2 = theta[num_parameters_1+1]
        elif num_parameters_2 == 3:
            rm_2 = theta[num_parameters_1+1]
            k_theta_2 = theta[num_parameters_1+2]
        elif num_parameters_2 == 4:
            rm_2 = theta[num_parameters_1+1]
            k_theta_2 = theta[num_parameters_1+2]
            mu_theta_2 = theta[num_parameters_1+3]
    for i in range(num_evaluations):
        contribution_1 = final_intensity_spheroid(I_porod,combinedfactor_1,distribution_1,formfactor_1,volume_1,Ibg,list_q[i],rm_1,C,sigma=sigma_rm_1,k=k_theta_1,mu=mu_theta_1)
        contribution_2 = nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,list_q[i],rm_2,sigma=sigma_rm_2,k=k_theta_2,mu=mu_theta_2)
        model_prediction[i] = contribution_1+contribution_2
    chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)

    return chi2

def chi2_nano_intensity_spheroid_double(theta, distribution_1, distribution_2, formfactor_1, formfactor_2 ,volume_1, volume_2,list_q,experiment,uncertainty):
    num_parameters = theta.size #record number of columns
    num_evaluations= list_q.size
    #check model for each distribution
    if callable(distribution_1):
        if formfactor_1 is formfactor_sphere:
            num_parameters_1 = 3
        elif formfactor_1 is formfactor_ellipsoid:
            num_parameters_1 = 4
        elif formfactor_1 is formfactor_coreshell:
            num_parameters_1 = 5
    else:
        if formfactor_1 is formfactor_sphere:
            num_parameters_1 = 2
        elif formfactor_1 is formfactor_ellipsoid:
            num_parameters_1 = 3
        elif formfactor_1 is formfactor_coreshell:
            num_parameters_1 = 4
    if callable(distribution_2):
        if formfactor_2 is formfactor_sphere:
            num_parameters_2 = 3
        elif formfactor_2 is formfactor_ellipsoid:
            num_parameters_2 = 4
        elif formfactor_2 is formfactor_coreshell:
            num_parameters_2 = 5
    else:
        if formfactor_2 is formfactor_sphere:
            num_parameters_2 = 2
        elif formfactor_2 is formfactor_ellipsoid:
            num_parameters_2 = 3
        elif formfactor_2 is formfactor_coreshell:
            num_parameters_2 = 4
    # logic for different cases
    combinedfactor_1 = np.exp(-theta[0])
    combinedfactor_2 = np.exp(-theta[num_parameters_1])
    sigma_rm_1 = None
    sigma_rm_2 = None
    k_theta_1 = None
    k_theta_2 = None
    mu_theta_1 = None
    mu_theta_2 = None
    model_prediction = np.zeros(num_evaluations)
    if callable(distribution_1):
        if num_parameters_1 == 3:
            rm_1 = theta[1]
            sigma_rm_1 = theta[1]*theta[2]
        elif num_parameters_1 == 4:
            rm_1 = theta[1]
            sigma_rm_1 = theta[1]*theta[2]
            k_theta_1 = theta[3]
        elif num_parameters_1 == 5:
            rm_1 = theta[1]
            sigma_rm_1 = theta[1]*theta[2]
            k_theta_1 = theta[3]
            mu_theta_1 = theta[4]
    else:
        if num_parameters_1 == 2:
            rm_1 = theta[1]
        elif num_parameters_1 == 3:
            rm_1 = theta[1]
            k_theta_1 = theta[2]
        elif num_parameters_1 == 4:
            rm_1 = theta[1]
            k_theta_1 = theta[2]
            mu_theta_1 = theta[3]
    if callable(distribution_2):
        if num_parameters_2 == 3:
            rm_2 = theta[num_parameters_1+1]
            sigma_rm_2 = theta[num_parameters_1+1]*theta[num_parameters_1+2]
        elif num_parameters_2 == 4:
            rm_2 = theta[num_parameters_1+1]
            sigma_rm_2 = theta[num_parameters_1+1]*theta[num_parameters_1+2]
            k_theta_2 = theta[num_parameters_1+3]
        elif num_parameters_2 == 5:
            rm_2 = theta[num_parameters_1+1]
            sigma_rm_2 = theta[num_parameters_1+1]*theta[num_parameters_1+2]
            k_theta_2 = theta[num_parameters_1+3]
            mu_theta_2 = theta[num_parameters_1+4]
    else:
        if num_parameters_2 == 2:
            rm_2 = theta[num_parameters_1+1]
        elif num_parameters_2 == 3:
            rm_2 = theta[num_parameters_1+1]
            k_theta_2 = theta[num_parameters_1+2]
        elif num_parameters_2 == 4:
            rm_2 = theta[num_parameters_1+1]
            k_theta_2 = theta[num_parameters_1+2]
            mu_theta_2 = theta[num_parameters_1+3]
    for i in range(num_evaluations):
        contribution_1 = nano_intensity_spheroid(combinedfactor_1,distribution_1,formfactor_1,volume_1,list_q[i],rm_1,sigma=sigma_rm_1,k=k_theta_1,mu=mu_theta_1)
        contribution_2 = nano_intensity_spheroid(combinedfactor_2,distribution_2,formfactor_2,volume_2,list_q[i],rm_2,sigma=sigma_rm_2,k=k_theta_2,mu=mu_theta_2)
        model_prediction[i] = contribution_1+contribution_2
    chi2 = loss_chi2(experiment,uncertainty,model_prediction,num_parameters)

    return chi2