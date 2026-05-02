"""
test file with manual input for checking validity of equations, optimization methods, etc. 
Essentially a functional main.py without ui
"""
from problems.coreshell import formfactor_coreshell
from problems.distributions import distribution_normal,distribution_lognormal
from problems.ellipsoid import volume_ellipsoid,formfactor_ellipsoid
from problems.sphere import volume_sphere,formfactor_sphere
from problems.approximations import I_porod
from core.models.assembled_problem import final_intensity_spheroid,nano_intensity_spheroid,double_intensity_spheroid,single_intensity_spheroid,double_intensity_spheroid_mag,single_intensity_spheroid_mag
from core.models.BO_model import chi2_final_intensity_spheroid_BO_single,chi2_final_intensity_spheroid_BO_double,chi2_final_intensity_spheroid_BO_double_with_reward,chi2_final_intensity_spheroid_BO_single_with_reward,chi2_nano_intensity_spheroid_BO_double,chi2_nano_intensity_spheroid_BO_single
from core.models.BO_model_fixed import chi2_final_intensity_spheroid_BO_double_fixed,chi2_final_intensity_spheroid_BO_single_fixed,chi2_nano_intensity_spheroid_BO_double_fixed,chi2_nano_intensity_spheroid_BO_single_fixed
from core.models.loss_functions import loss_chi2,make_residual, make_residual_nostop, TargetChi2Reached,make_joint_residuals,make_residuals_from_slice,make_weighted_joint_residuals
from core.utils.file_reader import file_reader_1d,file_reader_2d,file_reader_1d_nofilter
from core.acquisition.acquisition_functions import get_acq_qLogEI,get_acq_LogEI,build_model,build_model_nonoise
from core.utils.helper_functions import objective,chi2_red_variance,log_chi2_red_variance,joint_log_chi2_red_variance,construct_joint_bounds,construct_joint_parameters,separate_nuc_mag_parameters,assemble_theta,separate_nuc_mag_parameters_tensor
from core.optimizer.Levenberg_Marquardt import LM_optimize,LM_joint_optimize
from core.models.single_loss import chi2_final_intensity_spheroid_double,chi2_final_intensity_spheroid_single,chi2_nano_intensity_spheroid_double,chi2_nano_intensity_spheroid_single


from scipy.integrate import quad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize
from botorch.acquisition.analytic import LogNoisyExpectedImprovement,LogExpectedImprovement
import warnings
import time
import copy
import math



torch.set_default_dtype(torch.double)
torch.manual_seed(0)

#BO loop settings
max_iteration_BO = 75 #maximum BO loop number
max_iteration_BO_Pareto = 50 #maximum pareto sequential BO loop number
#reduced chi2 transform options
target_chi2 = 1
width_chi2 = 0.3

###input parameters
distribution_type = "log normal" #mono,normal,log normal,double
if distribution_type == "double":
    distribution_type_1 = "log normal" #mono,normal,log normal
    distribution_type_2 = "log normal" #mono,normal,log normal
    distribution_weight = 0.5 #weight of the first distribution function

model_1 = "sphere" #sphere, ellipsoid, core-shell
model_2 = "sphere" #sphere, ellipsoid, core-shell

### Start Parameters
log_Ibg = -6.29
log_C = -7.215
"""
Log_Ibg = -1.8736 
log_C = -3.1409
"""
#first distribution parameters
A_1 = 6.35
Rm_1 = 2.391
sigma_1 = 0.265
k_1 = 0.5
mu_1 = 0

"""
A_1 = 2.54
Rm_1 = 2.009
sigma_1 = 0.3
k_1 = 0.5
mu_1 = 0
"""
#second distribution parameters
A_2 = 2.9
Rm_2 = 7.9
sigma_2 = 0.3
k_2 = 0.5
mu_2 = 0


###mag scattering start parameters
### Start Parameters
log_Ibg_mag = None
log_C_mag = None
#log_Ibg_mag = -10
#log_C_mag = -10
"""
Log_Ibg = -1.8736 lo
log_C = -3.1409
"""
#first distribution parameters
A_1_mag = 4.359
Rm_1_mag = 2.617
sigma_1_mag = 0.204
k_1_mag = 0.5
mu_1_mag = 0

"""
A_1 = 2.54
Rm_1 = 2.009
sigma_1 = 0.3
k_1 = 0.5
mu_1 = 0
"""
#second distribution parameters
A_2_mag = 2.9
Rm_2_mag = 7.9
sigma_2_mag = 0.3
k_2_mag = 0.5
mu_2_mag = 0

###range settings
#hard-set limits
range_A = [0.1,10] #combined factor
range_Rm = [0.1,20] 
range_C = [-10,1] #approximation parameter
range_Ibg = [-10,1]
range_sigma = [0.01,0.3]
range_kellipsoid = [0.5,3]
range_kshell = [0,1]
range_mu = [-10,10]

#user-related bound ranges
search_width_A = 2
search_width_Rm = 4
search_width_C = 3
search_width_Ibg = 3
search_width_sigma = 0.1
search_width_kellipsoid = 0.5
search_width_kshell = 0.3
search_width_mu = 4

#first jointfit bound ranges
search_width_A_jointfit = 1
search_width_Rm_jointfit = 1
search_width_C_jointfit = 1
search_width_Ibg_jointfit = 1
search_width_sigma_jointfit = 0.1
search_width_kellipsoid_jointfit = 0.5
search_width_kshell_jointfit = 0.3
search_width_mu_jointfit = 1

#pareto slender bound ranges
search_width_A_jointfit_fine = 0.5
search_width_Rm_jointfit_fine = 0.5
search_width_C_jointfit_fine = 0.5
search_width_Ibg_jointfit_fine = 0.5
search_width_sigma_jointfit_fine = 0.1
search_width_kellipsoid_jointfit_fine = 0.5
search_width_kshell_jointfit_fine = 0.3
search_width_mu_jointfit_fine = 0.5

#convert to model parameters
combinedfactor_1 = np.exp(-A_1)
sigma_rm_1 = sigma_1*Rm_1
combinedfactor_2 = np.exp(-A_2)
sigma_rm_2 = sigma_2*Rm_2
Ibg = np.exp(log_Ibg)
C = np.exp(log_C)


#load data from txt file
test_Q, test_I, test_sigmaI, test_sigmaQ, test_Imag, test_sigmaImag = file_reader_2d("data\\27940.txt") #1d test case
tensor_I = torch.tensor(test_I)
tensor_sigmaI = torch.tensor(test_sigmaI)
print(tensor_sigmaI.shape)
print(test_sigmaImag.shape)
#print(test_Q)
#log_test_I = np.log(test_I)
###computation of porod and background scattering signal
num_parameters = 0
#case by case initialization
if (distribution_type == "normal") or (distribution_type == "log normal") :
    if model_1 == "sphere":
        num_parameters_1 = 5
    elif model_1 == "ellipsoid":
        num_parameters_1 = 6
    elif model_1 == "core-shell":
        num_parameters_1 = 7
    num_parameters = num_parameters_1
elif distribution_type == "mono":
    if model_1 == "sphere":
        num_parameters_1 = 4
    elif model_1 == "ellipsoid":
        num_parameters_1 = 5
    elif model_1 == "core-shell":
        num_parameters_1 = 6
    num_parameters = num_parameters_1
elif distribution_type == "double":
    if (distribution_type_1 == "normal") or (distribution_type_1 == "log normal") :
        if model_1 == "sphere":
            num_parameters_1 = 5
        elif model_1 == "ellipsoid":
            num_parameters_1 = 6
        elif model_1 == "core-shell":
            num_parameters_1 = 7
    elif distribution_type_1 == "mono":
        if model_1 == "sphere":
            num_parameters_1 = 4
        elif model_1 == "ellipsoid":
            num_parameters_1 = 5
        elif model_1 == "core-shell":
            num_parameters_1 = 6
    if (distribution_type_2 == "normal") or (distribution_type_2 == "log normal") :
        if model_2 == "sphere":
            num_parameters_2 = 3
        elif model_2 == "ellipsoid":
            num_parameters_2 = 4
        elif model_2 == "core-shell":
            num_parameters_2 = 5
    elif distribution_type_2 == "mono":
        if model_2 == "sphere":
            num_parameters_2 = 2
        elif model_2 == "ellipsoid":
            num_parameters_2 = 3
        elif model_2 == "core-shell":
            num_parameters_2 = 4
    num_parameters = num_parameters_1+num_parameters_2


#model_Q = np.linspace(0.01,3,100)
#model_I = [(final_intensity_spheroid(I_porod,combinedfactor_1,distribution_lognormal,formfactor_sphere,volume_sphere,Ibg,q,Rm_1,C,sigma=sigma_rm_1)+nano_intensity_spheroid(combinedfactor_2,distribution_lognormal,formfactor_sphere,volume_sphere,q,Rm_2,sigma=sigma_rm_2)) for q in model_Q]
#model_I = [final_intensity_spheroid(I_porod,combinedfactor_1,distribution_lognormal,formfactor_sphere,volume_sphere,Ibg,q,Rm_1,C,sigma=sigma_rm_1) for q in model_Q]
"""
###
#plot initial case
fig = go.Figure()

fig.add_trace(go.Scatter(
    x = test_Q,
    y = np.log(test_I),
    mode = "markers",
    name = "Experimental",
    marker = dict(size=10)
))
fig.add_trace(go.Scatter(
    x=model_Q,
    y=np.log(model_I),
    mode="lines",
    name = "Model"
))
fig.update_layout(
    title = "example plot",
    xaxis_title = "Q(nm-1)",
    yaxis_title = "log(I(Q))(cm-1)",
    template = "plotly_white"
)

fig.show()
###
"""
### BO iteration ###

#initialization
#bounds
bounds_1 = torch.tensor([[max(range_Ibg[0],log_Ibg-search_width_Ibg),min(range_Ibg[1],log_Ibg+search_width_Ibg)],
                         [max(range_C[0],log_C-search_width_C),min(range_C[1],log_C+search_width_C)],
                         [max(range_A[0],A_1-search_width_A),min(range_A[1],A_1+search_width_A)],
                         [max(range_Rm[0],Rm_1-search_width_Rm),min(range_Rm[1],Rm_1+search_width_Rm)]])
bounds_box_1 = [range_Ibg,range_C,range_A,range_Rm]
joint_searchwidth_1 = [search_width_Ibg_jointfit,search_width_C_jointfit,search_width_A_jointfit,search_width_Rm_jointfit]
joint_searchwidth_1_fine = [search_width_Ibg_jointfit_fine,search_width_C_jointfit_fine,search_width_A_jointfit_fine,search_width_Rm_jointfit_fine]
bounds_2 = torch.tensor([[max(range_A[0],A_2-search_width_A),min(range_A[1],A_2+search_width_A)],
                         [max(range_Rm[0],Rm_2-search_width_Rm),min(range_Rm[1],Rm_2+search_width_Rm)]])
bounds_box_2 = [range_A,range_Rm]
joint_searchwidth_2 = [search_width_A_jointfit,search_width_Rm_jointfit]
joint_searchwidth_2_fine = [search_width_A_jointfit_fine,search_width_Rm_jointfit_fine]
#mag signal bounds
if log_Ibg_mag is not None:
    bounds_1_mag = torch.tensor([[max(range_Ibg[0],log_Ibg_mag-search_width_Ibg),min(range_Ibg[1],log_Ibg_mag+search_width_Ibg)],
                            [max(range_C[0],log_C_mag-search_width_C),min(range_C[1],log_C_mag+search_width_C)],
                            [max(range_A[0],A_1_mag-search_width_A),min(range_A[1],A_1_mag+search_width_A)],
                            [max(range_Rm[0],Rm_1_mag-search_width_Rm),min(range_Rm[1],Rm_1_mag+search_width_Rm)]])
    params_mag_1 = ['background_log_mag','C_log_mag','combinedfactor_log_mag']
else:
    bounds_1_mag = torch.tensor([[max(range_A[0],A_1_mag-search_width_A),min(range_A[1],A_1_mag+search_width_A)],
                            [max(range_Rm[0],Rm_1_mag-search_width_Rm),min(range_Rm[1],Rm_1_mag+search_width_Rm)]])
    params_mag_1 = ['combinedfactor_log_mag']
bounds_2_mag = torch.tensor([[max(range_A[0],A_2_mag-search_width_A),min(range_A[1],A_2_mag+search_width_A)],
                        [max(range_Rm[0],Rm_2_mag-search_width_Rm),min(range_Rm[1],Rm_2_mag+search_width_Rm)]])

#list of parameters to optimize
params_1 = ['background_log','C_log','combinedfactor_log','R']
params_2 = ['combinedfactor_2_log','R_2']
common_params = ['R']
possible_common_params_single = ['R','sigma','k']
possible_common_params_double = ['R','sigma','k','R_2','sigma_2','k_2']
#list of fixed parameters
fixed_kwargs = {'approx': I_porod}
# assign distribution function
distribution_1 = "mono"
distribution_2 = "mono"
if distribution_type == "log normal":
    distribution_1 = distribution_lognormal
    bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
    bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
    params_1.append('sigma')
    common_params.append('sigma')
    bounds_box_1.append(range_sigma)
    joint_searchwidth_1.append(search_width_sigma_jointfit)
    joint_searchwidth_1_fine.append(search_width_sigma_jointfit_fine)
elif distribution_type == "normal":
    distribution_1 = distribution_normal
    bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
    bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
    params_1.append('sigma')
    common_params.append('sigma')
    bounds_box_1.append(range_sigma)
    joint_searchwidth_1.append(search_width_sigma_jointfit)
    joint_searchwidth_1_fine.append(search_width_sigma_jointfit_fine)
elif distribution_type == "double":
    common_params.append('R_2')
    if distribution_type_1 == "log normal":
        distribution_1 = distribution_lognormal
        bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
        bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
        params_1.append('sigma')
        common_params.append('sigma')
        bounds_box_1.append(range_sigma)
        joint_searchwidth_1.append(search_width_sigma_jointfit)
        joint_searchwidth_1_fine.append(search_width_sigma_jointfit_fine)
    elif distribution_type_1 == "normal":
        distribution_1 = distribution_normal
        bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
        bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
        params_1.append('sigma')
        common_params.append('sigma')
        bounds_box_1.append(range_sigma)
        joint_searchwidth_1.append(search_width_sigma_jointfit)
        joint_searchwidth_1_fine.append(search_width_sigma_jointfit_fine)
    if distribution_type_2 == "log normal":
        distribution_2 = distribution_lognormal
        bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_sigma[0],sigma_2-search_width_sigma),min(range_sigma[1],sigma_2+search_width_sigma)]])])
        bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_sigma[0],sigma_2_mag-search_width_sigma),min(range_sigma[1],sigma_2_mag+search_width_sigma)]])])
        params_2.append('sigma_2')
        common_params.append('sigma_2')
        bounds_box_2.append(range_sigma)
        joint_searchwidth_2.append(search_width_sigma_jointfit)
        joint_searchwidth_2_fine.append(search_width_sigma_jointfit_fine)
    elif distribution_type_2 == "normal":
        distribution_2 = distribution_normal
        bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_sigma[0],sigma_2-search_width_sigma),min(range_sigma[1],sigma_2+search_width_sigma)]])])
        bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_sigma[0],sigma_2_mag-search_width_sigma),min(range_sigma[1],sigma_2_mag+search_width_sigma)]])])
        params_2.append('sigma_2')
        common_params.append('sigma_2')
        bounds_box_2.append(range_sigma)
        joint_searchwidth_2.append(search_width_sigma_jointfit)
        joint_searchwidth_2_fine.append(search_width_sigma_jointfit_fine)
fixed_kwargs['distribution'] = distribution_1
if distribution_type == "double":
    fixed_kwargs['distribution_2'] = distribution_2



#assign models
if model_1 == "sphere":
    formfactor_1 = formfactor_sphere
    volume_1 = volume_sphere
elif model_1 == "ellipsoid":
    formfactor_1 = formfactor_ellipsoid
    volume_1 = volume_ellipsoid
    bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_kellipsoid[0],k_1-search_width_kellipsoid),min(range_kellipsoid[1],k_1+search_width_kellipsoid)]])])
    bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_kellipsoid[0],k_1_mag-search_width_kellipsoid),min(range_kellipsoid[1],k_1_mag+search_width_kellipsoid)]])])
    params_1.append('k')
    common_params.append('k')
    bounds_box_1.append(range_kellipsoid)
    joint_searchwidth_1.append(search_width_kellipsoid_jointfit)
    joint_searchwidth_1_fine.append(search_width_kellipsoid_jointfit_fine)
elif model_1 == "core-shell":
    formfactor_1 = formfactor_coreshell
    volume_1 = volume_sphere
    bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_kshell[0],k_1-search_width_kshell),min(range_kshell[1],k_1+search_width_kshell)]])])
    bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_mu[0],mu_1-search_width_mu),min(range_mu[1],mu_1+search_width_mu)]])])
    bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_kshell[0],k_1_mag-search_width_kshell),min(range_kshell[1],k_1_mag+search_width_kshell)]])])
    bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_mu[0],mu_1_mag-search_width_mu),min(range_mu[1],mu_1_mag+search_width_mu)]])])
    params_1.append('k')
    common_params.append('k')
    bounds_box_1.append(range_kshell)
    joint_searchwidth_1.append(search_width_kshell_jointfit)
    joint_searchwidth_1_fine.append(search_width_kshell_jointfit_fine)
    params_1.append('mu')
    bounds_box_1.append(range_mu)
    joint_searchwidth_1.append(search_width_mu_jointfit)
    joint_searchwidth_1_fine.append(search_width_mu_jointfit_fine)


if model_2 == "sphere":
    formfactor_2 = formfactor_sphere
    volume_2 = volume_sphere
elif model_2 == "ellipsoid":
    formfactor_2 = formfactor_ellipsoid
    volume_2 = volume_ellipsoid
    bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_kellipsoid[0],k_2-search_width_kellipsoid),min(range_kellipsoid[1],k_2+search_width_kellipsoid)]])])
    bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_kellipsoid[0],k_2_mag-search_width_kellipsoid),min(range_kellipsoid[1],k_2_mag+search_width_kellipsoid)]])])
    params_2.append('k_2')
    common_params.append('k_2')
    bounds_box_2.append(range_kellipsoid)
    joint_searchwidth_2.append(search_width_kellipsoid_jointfit)
    joint_searchwidth_2_fine.append(search_width_kellipsoid_jointfit_fine)
elif model_2 == "core-shell":
    formfactor_2 = formfactor_coreshell
    volume_2 = volume_sphere
    bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_kshell[0],k_2-search_width_kshell),min(range_kshell[1],k_2+search_width_kshell)]])])
    bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_mu[0],mu_2-search_width_mu),min(range_mu[1],mu_2+search_width_mu)]])])
    bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_kshell[0],k_2_mag-search_width_kshell),min(range_kshell[1],k_2_mag+search_width_kshell)]])])
    bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_mu[0],mu_2_mag-search_width_mu),min(range_mu[1],mu_2_mag+search_width_mu)]])])
    params_2.append('k_2')
    common_params.append('k_2')
    bounds_box_2.append(range_kshell)
    joint_searchwidth_2.append(search_width_kshell_jointfit)
    joint_searchwidth_2_fine.append(search_width_kshell_jointfit_fine)
    params_2.append('mu_2')
    bounds_box_2.append(range_mu)
    joint_searchwidth_2.append(search_width_mu_jointfit)
    joint_searchwidth_2_fine.append(search_width_mu_jointfit_fine)

fixed_kwargs['distribution'] = distribution_1
fixed_kwargs['formfactor'] = formfactor_1
fixed_kwargs['volume'] = volume_1
if distribution_type == "double":
    fixed_kwargs['distribution_2'] = distribution_2
    fixed_kwargs['formfactor_2'] = formfactor_2
    fixed_kwargs['volume_2'] = volume_2
pos_array_A_2 = -1
pos_array_mu_1 = -1
pos_array_mu_2 = -1
if log_Ibg_mag is not None:
    nuc_pos_list = [0,1,2] #corresponds to Ibg, C, A, or only A if porod and bg is disabled
    mag_pos_list = [0,1,2]
else:
    nuc_pos_list = [2]
    mag_pos_list = [0]
#assign BO bounds
if distribution_type == "double":
    bounds = torch.cat([bounds_1,bounds_2])
    bounds_mag = torch.cat([bounds_1_mag,bounds_2_mag])
    bounds = bounds.T
    bounds_mag = bounds_mag.T
    params = params_1 + params_2
    bounds_box_np_1 = np.array(bounds_box_1)
    joint_searchwidth = joint_searchwidth_1 + joint_searchwidth_2
    joint_searchwidth_fine = joint_searchwidth_1_fine + joint_searchwidth_2_fine
    bounds_box_np_2 = np.array(bounds_box_2)
    pos_array_A_2 = bounds_box_np_1.shape[0] # notes relative position of combined factor 1 in parameter set
    nuc_pos_list.append(pos_array_A_2)
    mag_pos_list.append(mag_pos_list[-1]+1)
    params_mag_1.append('combinedfactor_2_log_mag')
    if model_1 == "core-shell":
        pos_array_mu_1 = bounds_box_np_1.shape[0]-1 
        nuc_pos_list.append(pos_array_mu_1)
        mag_pos_list.append(mag_pos_list[-1]+1)
        params_mag_1.append('mu_mag')
    if model_2 == "core-shell":
        pos_array_mu_2 = bounds_box_np_1.shape[0] + bounds_box_np_2.shape[0]-1
        nuc_pos_list.append(pos_array_mu_2)
        mag_pos_list.append(mag_pos_list[-1]+1)
        params_mag_1.append('mu_2_mag')
    bounds_box_np = np.vstack([bounds_box_np_1,bounds_box_np_2])
else:
    bounds = bounds_1.T
    bounds_mag = bounds_1_mag.T
    params = params_1
    bounds_box_np = np.array(bounds_box_1)
    joint_searchwidth = joint_searchwidth_1
    joint_searchwidth_fine = joint_searchwidth_1_fine
    if model_1 == "core-shell":
        pos_array_mu_1 = bounds_box_np.shape[0]-1
        nuc_pos_list.append(pos_array_mu_1)
        mag_pos_list.append(mag_pos_list[-1]+1)
        params_mag_1.append('mu_mag')

params_mag = params_mag_1 + common_params
jointfit_params = params + params_mag_1
#beta = 3.0/(1/bounds.shape[1]**0.5)
print("jointfit_params:", jointfit_params)
print("fixed_kwargs:", fixed_kwargs)
print("params:", params)
print("params_mag:", params_mag)
beta = 6 #smaller beta makes search more aggressive
dof = test_I.shape[0]-bounds.shape[1] #degree of freedom for chi2
if log_Ibg_mag is not None:
    dof_mag = dof
else:
    dof_mag = dof + 2
#initial guess values
n_init = 100
"""
t0 = time.time()
X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, bounds.shape[1])
X_mag = bounds_mag[0] + (bounds_mag[1]-bounds_mag[0])*torch.rand(n_init, bounds_mag.shape[1])
#print(X.shape)
#print(X.shape[-1])
#print(bounds.shape)
#print(X)

if distribution_type == "double":
    Y = chi2_final_intensity_spheroid_BO_double(X,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
    if log_Ibg_mag is not None:
        Y_mag = chi2_final_intensity_spheroid_BO_double(X_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        Y_mag = chi2_nano_intensity_spheroid_BO_double(X_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
else:
    Y = chi2_final_intensity_spheroid_BO_single(X,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
    if log_Ibg_mag is not None:
        Y_mag = chi2_final_intensity_spheroid_BO_single(X_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    else:
        Y_mag = chi2_nano_intensity_spheroid_BO_single(X_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
   


#if distribution_type == "double":
#    Y,reward_variance = chi2_final_intensity_spheroid_BO_double_with_reward(X,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI,target_chi2,width_chi2)
#else:
#    Y,reward_variance = chi2_final_intensity_spheroid_BO_single_with_reward(X,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI,target_chi2,width_chi2)
Y = Y.unsqueeze(-1)
#Y_variance = log_chi2_red_variance(Y,dof)
#reward_variance = reward_variance.unsqueeze(-1)

Y_mag = Y_mag.unsqueeze(-1)

#print(Y)

t1 = time.time()
print(f"initial random guess took {t1-t0:.1f}s")
#print(X.shape)        # should be (50, 5)
#print(Y.shape)        # should be (50, 1) — note: must be 2D
#print(torch.isnan(Y).any())   # must be False
#print(torch.isinf(Y).any())   # must be False
#print(Y.dtype)        # should be float32 or float64
#print(Y)
#BO loop

log_Y = torch.log(Y)
log_Y_mag = torch.log(Y_mag)


for iteration in range(max_iteration_BO):
    #with warnings.catch_warnings():
    #    warnings.simplefilter('error', category=RuntimeWarning)
    #    try: 
            #print(Y_variance.min(), Y_variance.max())
            t0 = time.time()
            s2_Y = log_Y.var()
            Y_variance = log_chi2_red_variance(Y,dof,s2=s2_Y)
            model_gp = build_model(X,log_Y,Y_variance,beta)
            
            #print(s2_Y)
            #print((Y_variance / s2_Y).min(), (Y_variance / s2_Y).max())
            acq = LogNoisyExpectedImprovement(model_gp,X,maximize=False)
            #acq = get_acq_LogEI(X,Y,Y_variance,beta)
            t1 = time.time()
            print(f"get_acq took {t1-t0:.1f}s")
            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            t2 = time.time()
            print(f"optimize_acqf took {t2-t1:.1f}s")
            if distribution_type == "double":
                new_Y = chi2_final_intensity_spheroid_BO_double(candidate,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
            else:
                new_Y = chi2_final_intensity_spheroid_BO_single(candidate,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
            t3 = time.time()
            print(f"chi2 took {t3-t2:.1f}s")
            new_Y = new_Y.unsqueeze(-1)
            new_log_Y = torch.log(new_Y)
            #new_variance = log_chi2_red_variance(new_Y,dof,s2=s2_Y)
            # Update dataset
            X = torch.cat([X, candidate])
            Y = torch.cat([Y, new_Y])
            log_Y = torch.cat([log_Y,new_log_Y])
            
            #Y_variance = torch.cat([Y_variance,new_variance])
            print(f"Iteration {iteration}: best Y = {Y.min().item():.4f}, new candidate = {candidate}")
            condition = (Y >= 0.9) & (Y <= 2.0)
            count = torch.sum(condition).item()
            if count >= 3:
                #indices = torch.nonzero(condition)
                print(f"Acceptable fit: chi2_red = {Y.min().item():.3f}")
                break
    #    except RuntimeWarning as e:
    #        print(f"BO loop Warning at iteration {iteration}: {e}")
best_idx = torch.argmin(Y)
print("Best theta:", X[best_idx])
print("Best chi2:", Y[best_idx])

if torch.any(condition).item(): 
    condition = condition.squeeze(1)
    #indices = torch.nonzero(condition)
    startpoints = X[condition]
    if count != 1:
        BO_candidates = startpoints.detach().cpu().numpy()
    else:
        BO_candidates = [X[best_idx].detach().cpu().numpy()]
else:
    BO_candidates = [X[best_idx].detach().cpu().numpy()]
    #BO_candidates.reshape(1,bounds.shape[1])
print("BO_candidates:", BO_candidates)
if distribution_type == "double":
    result = LM_optimize(double_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params=params,kwargs=fixed_kwargs,startpoints=BO_candidates)
else:
    result = LM_optimize(single_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params=params,kwargs=fixed_kwargs,startpoints=BO_candidates)

print(f"Best result: Parameters = {result.x}, chi2_red = {result.cost*2/dof:.4f}")

J = result.jac #jacobian
H_approx = J.T @ J #hessian approximation
cov = np.linalg.inv(H_approx) #covariance matrix
uncertainties = np.sqrt(np.diag(cov)) #fit uncertainties


### Now do the same for magnetic scattering ###
for iteration in range(max_iteration_BO):
    #with warnings.catch_warnings():
    #    warnings.simplefilter('error', category=RuntimeWarning)
    #    try: 
            #print(Y_variance.min(), Y_variance.max())
            t0 = time.time()
            s2_Y_mag = log_Y_mag.var()
            Y_mag_variance = log_chi2_red_variance(Y_mag,dof,s2=s2_Y_mag)
            model_gp_mag = build_model(X_mag,log_Y_mag,Y_mag_variance,beta)
            
            #print(s2_Y)
            #print((Y_variance / s2_Y).min(), (Y_variance / s2_Y).max())
            acq_mag = LogNoisyExpectedImprovement(model_gp_mag,X_mag,maximize=False)
            #acq = get_acq_LogEI(X,Y,Y_variance,beta)
            t1 = time.time()
            print(f"get_acq took {t1-t0:.1f}s")
            candidate_mag, _ = optimize_acqf(
                acq_function=acq_mag,
                bounds=bounds_mag,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            t2 = time.time()
            print(f"optimize_acqf took {t2-t1:.1f}s")
            if distribution_type == "double":
                if log_Ibg_mag is not None:
                    new_Y_mag = chi2_final_intensity_spheroid_BO_double(candidate_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
                else:
                    new_Y_mag = chi2_nano_intensity_spheroid_BO_double(candidate_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                if log_Ibg_mag is not None:
                    new_Y_mag = chi2_final_intensity_spheroid_BO_single(candidate_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
                else:
                    new_Y_mag = chi2_nano_intensity_spheroid_BO_single(candidate_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            t3 = time.time()
            print(f"chi2 took {t3-t2:.1f}s")
            new_Y_mag = new_Y_mag.unsqueeze(-1)
            new_log_Y_mag = torch.log(new_Y_mag)
            #new_variance_mag = log_chi2_red_variance(new_Y_mag,dof,s2=s2_Y_mag)
            # Update dataset
            X_mag = torch.cat([X_mag, candidate_mag])
            Y_mag = torch.cat([Y_mag, new_Y_mag])
            log_Y_mag = torch.cat([log_Y_mag,new_log_Y_mag])
            
            #Y_mag_variance = torch.cat([Y_mag_variance,new_variance_mag])
            print(f"Iteration {iteration}: best Ymag = {Y_mag.min().item():.4f}, new candidate = {candidate_mag}")
            condition_mag = (Y_mag >= 0.9) & (Y_mag <= 2.0)
            count = torch.sum(condition_mag).item()
            if count >= 3:
                #indices_mag = torch.nonzero(condition_mag)
                print(f"Acceptable fit: chi2_red = {Y_mag.min().item():.3f}")
                break
    #    except RuntimeWarning as e:
    #        print(f"BO loop Warning at iteration {iteration}: {e}")
best_idx_mag = torch.argmin(Y_mag)
print("Best theta mag:", X_mag[best_idx_mag])
print("Best chi2 mag:", Y_mag[best_idx_mag])

if torch.any(condition_mag).item():
    condition_mag = condition_mag.squeeze(1)
    #indices_mag = torch.nonzero(condition_mag)
    startpoints_mag = X_mag[condition_mag]
    if count != 1:
        BO_candidates_mag = startpoints_mag.detach().cpu().numpy()
    else:
        BO_candidates_mag = [X_mag[best_idx_mag].detach().cpu().numpy()]
else:
    BO_candidates_mag = [X_mag[best_idx_mag].detach().cpu().numpy()]
    #BO_candidates_mag.reshape(1,bounds_mag.shape[1])

#BO_candidates_mag = [[4.32468673, 2.52696484, 0.21628018]]
#print("BO_candidates_mag:", BO_candidates_mag)
#print(fixed_kwargs)
#print(params)
if log_Ibg_mag is not None:
    params_firstfit_mag = params
else:
    params_firstfit_mag = copy.deepcopy(params)
    params_firstfit_mag.pop(0)
    params_firstfit_mag.pop(0)
    print("params_firstfit_mag:", params_firstfit_mag)
if distribution_type == "double":
    result_mag = LM_optimize(double_intensity_spheroid,test_Q,'Q',test_Imag,test_sigmaImag,params_firstfit_mag,fixed_kwargs,BO_candidates_mag)
else:
    result_mag = LM_optimize(single_intensity_spheroid,test_Q,'Q',test_Imag,test_sigmaImag,params_firstfit_mag,fixed_kwargs,BO_candidates_mag)

print(f"Best result: Parameters = {result_mag.x}, chi2_red = {result_mag.cost*2/dof:.4f}")

J_mag = result_mag.jac #jacobian
H_approx_mag = J_mag.T @ J_mag #hessian approximation
cov_mag = np.linalg.inv(H_approx_mag) #covariance matrix
uncertainties_mag = np.sqrt(np.diag(cov_mag)) #fit uncertainties




### Joint fit ###
jointstart = np.array(result.x)
jointstart_mag = np.array(result_mag.x)
"""
jointstart = np.array([-6.82333048,-7.22396437,6.27986562,2.16129074,0.30854942])
jointstart_mag = np.array([4.37764956,2.70738054,0.18221415])
### resolve nuc and mag fitting with fixed params
n_init_fine = 50
fixed_args_nuc_complement = copy.deepcopy(fixed_kwargs)
fixed_args_mag_complement = copy.deepcopy(fixed_kwargs)
nuc_params_temp = np.array(params)
mag_params_temp = np.array(params_mag)
mask_params_nuc = ~np.isin(nuc_params_temp,common_params)
mask_params_nuc_torch = torch.tensor(mask_params_nuc)
mask_params_mag = ~np.isin(mag_params_temp,common_params)
mask_params_mag_torch = torch.tensor(mask_params_mag)
nuc_params_filtered = nuc_params_temp[mask_params_nuc].tolist()
bounds_nuc_filtered = bounds[:,mask_params_nuc_torch]
mag_params_filtered = mag_params_temp[mask_params_mag].tolist()
bounds_mag_filtered = bounds_mag[:,mask_params_mag_torch]
jointstart_nuc_filtered = jointstart[mask_params_nuc]
jointstart_mag_filtered = jointstart_mag[mask_params_mag]

mask_params_nuc_del = np.isin(nuc_params_temp,common_params)
mask_params_mag_del = np.isin(mag_params_temp,common_params)
deleted_nuc_params = nuc_params_temp[mask_params_nuc_del]
deleted_mag_params = mag_params_temp[mask_params_mag_del]
deleted_nuc_values = jointstart[mask_params_nuc_del]
deleted_mag_values = jointstart_mag[mask_params_mag_del]
#fill dummy values for dicts
if distribution_type == "double":
    for entry in possible_common_params_double:
        fixed_args_nuc_complement[entry] = None
        fixed_args_mag_complement[entry] = None
else:
    for entry in possible_common_params_single:
        fixed_args_nuc_complement[entry] = None
        fixed_args_mag_complement[entry] = None


for s,v in zip(deleted_mag_params,deleted_mag_values):# for nuc fixed fitting, use mag results, and vice versa
    fixed_args_nuc_complement[s] = v

for s,v in zip(deleted_nuc_params,deleted_nuc_values):
    fixed_args_mag_complement[s] = v
#print("jointfit_params:", jointfit_params)
#print("bounds tensor:", bounds)
#print("bounds tensor shape:", bounds.shape)
#print("nuc_params_filtered:",nuc_params_filtered)
#print("mag_params_filtered:", mag_params_filtered)
#print("jointstart_nuc_filtered:", jointstart_nuc_filtered)
#print("jointstart_mag_filtered:", jointstart_mag_filtered)
#print("fixed_args_nuc_complement:", fixed_args_nuc_complement)
#print("fixed_args_mag_complement:", fixed_args_mag_complement)
#print("bounds_nuc_filtered:", bounds_nuc_filtered)
#print("bounds_mag_filtered:", bounds_mag_filtered)
#do nuclear optimization starting from the fixed values

"""
X_filtered_nuc = bounds_nuc_filtered[0]+(bounds_nuc_filtered[1]-bounds_nuc_filtered[0])*torch.rand(n_init_fine,bounds_nuc_filtered.shape[1])
X_filtered_mag = bounds_mag_filtered[0]+(bounds_mag_filtered[1]-bounds_mag_filtered[0])*torch.rand(n_init_fine,bounds_mag_filtered.shape[1])
if distribution_type == "double":
    Y_filtered_nuc = chi2_final_intensity_spheroid_BO_double_fixed(X_filtered_nuc,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                   test_I,test_sigmaI,fixed_args_nuc_complement['R'],
                                                                   fixed_args_nuc_complement['R_2'],
                                                                   sigma_1=fixed_args_nuc_complement['sigma'],
                                                                   k_1=fixed_args_nuc_complement['k'],
                                                                   sigma_2=fixed_args_nuc_complement['sigma_2'],
                                                                   k_2=fixed_args_nuc_complement['k_2'])
    if log_Ibg_mag is not None:
        Y_filtered_mag = chi2_final_intensity_spheroid_BO_double_fixed(X_filtered_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                   test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                   fixed_args_mag_complement['R_2'],
                                                                   sigma_1=fixed_args_mag_complement['sigma'],
                                                                   k_1=fixed_args_mag_complement['k'],
                                                                   sigma_2=fixed_args_mag_complement['sigma_2'],
                                                                   k_2=fixed_args_mag_complement['k_2'])
    else:
        Y_filtered_mag = chi2_nano_intensity_spheroid_BO_double_fixed(X_filtered_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                   test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                   fixed_args_mag_complement['R_2'],
                                                                   sigma_1=fixed_args_mag_complement['sigma'],
                                                                   k_1=fixed_args_mag_complement['k'],
                                                                   sigma_2=fixed_args_mag_complement['sigma_2'],
                                                                   k_2=fixed_args_mag_complement['k_2'])
else:
    Y_filtered_nuc = chi2_final_intensity_spheroid_BO_single_fixed(X_filtered_nuc,distribution_1,formfactor_1,volume_1,test_Q,
                                                                   test_I,test_sigmaI,fixed_args_nuc_complement['R'],
                                                                   sigma=fixed_args_nuc_complement['sigma'],
                                                                   k=fixed_args_nuc_complement['k'])
    if log_Ibg_mag is not None:
        Y_filtered_mag = chi2_final_intensity_spheroid_BO_single_fixed(X_filtered_mag,distribution_1,formfactor_1,volume_1,test_Q,
                                                                   test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                   sigma=fixed_args_mag_complement['sigma'],
                                                                   k=fixed_args_mag_complement['k'])
    else:
        Y_filtered_mag = chi2_nano_intensity_spheroid_BO_single_fixed(X_filtered_mag,distribution_1,formfactor_1,volume_1,test_Q,
                                                                   test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                   sigma=fixed_args_mag_complement['sigma'],
                                                                   k=fixed_args_mag_complement['k'])
        
Y_filtered_nuc = Y_filtered_nuc.unsqueeze(-1)
Y_filtered_mag = Y_filtered_mag.unsqueeze(-1)

log_Y_filtered_nuc = torch.log(Y_filtered_nuc)
log_Y_filtered_mag = torch.log(Y_filtered_mag)

for iteration in range(max_iteration_BO_Pareto):
    model_gp = build_model_nonoise(X_filtered_nuc,log_Y_filtered_nuc,beta)
    best_f = log_Y_filtered_nuc.min().item()
    acq = LogExpectedImprovement(model_gp,best_f,maximize=False)#no need to add noise, the model is already imperfect enough
    candidate,_ = optimize_acqf(
        acq_function=acq,
        bounds = bounds_nuc_filtered,
        q = 1,
        num_restarts=10,
        raw_samples=256,
    )
    if distribution_type == "double":
        new_Y_filtered_nuc = chi2_final_intensity_spheroid_BO_double_fixed(candidate,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                    test_I,test_sigmaI,fixed_args_nuc_complement['R'],
                                                                    fixed_args_nuc_complement['R_2'],
                                                                    sigma_1=fixed_args_nuc_complement['sigma'],
                                                                    k_1=fixed_args_nuc_complement['k'],
                                                                    sigma_2=fixed_args_nuc_complement['sigma_2'],
                                                                    k_2=fixed_args_nuc_complement['k_2'])
    else:
        new_Y_filtered_nuc = chi2_final_intensity_spheroid_BO_single_fixed(candidate,distribution_1,formfactor_1,volume_1,test_Q,
                                                                    test_I,test_sigmaI,fixed_args_nuc_complement['R'],
                                                                    sigma=fixed_args_nuc_complement['sigma'],
                                                                    k=fixed_args_nuc_complement['k'])
    new_Y_filtered_nuc = new_Y_filtered_nuc.unsqueeze(-1)
    new_log_Y_filtered_nuc = torch.log(new_Y_filtered_nuc)
    X_filtered_nuc = torch.cat([X_filtered_nuc,candidate])
    Y_filtered_nuc = torch.cat([Y_filtered_nuc,new_Y_filtered_nuc])
    log_Y_filtered_nuc = torch.cat([log_Y_filtered_nuc,new_log_Y_filtered_nuc])
    print(f"Iteration {iteration}: best Y_filtered_nuc = {Y_filtered_nuc.min().item():.4f}, new candidate = {candidate}")
best_filtered_nuc_idx = torch.argmin(Y_filtered_nuc)
BO_filtered_nuc_candidate = [X_filtered_nuc[best_filtered_nuc_idx].detach().cpu().numpy()]
if distribution_type == "double":
    filtered_res_nuc = LM_optimize(double_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params=nuc_params_filtered,kwargs=fixed_args_nuc_complement,startpoints=BO_filtered_nuc_candidate)
else:
    filtered_res_nuc = LM_optimize(single_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params=nuc_params_filtered,kwargs=fixed_args_nuc_complement,startpoints=BO_filtered_nuc_candidate)
print(f"Complement result nuc: Parameters = {filtered_res_nuc.x}, chi2_red = {filtered_res_nuc.cost*2/dof:.4f}")

filtered_res_nuc_np = np.array(filtered_res_nuc.x)
assembled_theta_magside = assemble_theta(jointfit_params,nuc_params_filtered,filtered_res_nuc_np,mag_params_filtered,jointstart_mag_filtered,fixed_args_nuc_complement)
print("Lamda = 0 (only magnetic influence) fit result:", assembled_theta_magside)
"""
assembled_theta_magside = [-6.06404703,-7.18891623,6.41047672,2.70738054,0.18221415,4.37764956]
assembled_theta_magside_chi2_nuc = 1.2798
assembled_theta_magside_chi2_mag = 1.3487788062955082

### Now do the same for mag scattering

"""
for iteration in range(max_iteration_BO_Pareto):
    model_gp = build_model_nonoise(X_filtered_mag,log_Y_filtered_mag,beta)
    best_f = log_Y_filtered_mag.min().item()
    acq = LogExpectedImprovement(model_gp,best_f,maximize=False)#no need to add noise, the model is already imperfect enough
    candidate,_ = optimize_acqf(
        acq_function=acq,
        bounds = bounds_mag_filtered,
        q = 1,
        num_restarts=10,
        raw_samples=256,
    )
    if distribution_type == "double":
        if log_Ibg_mag is not None:
            new_Y_filtered_mag = chi2_final_intensity_spheroid_BO_double_fixed(candidate,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                    test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                    fixed_args_mag_complement['R_2'],
                                                                    sigma_1=fixed_args_mag_complement['sigma'],
                                                                    k_1=fixed_args_mag_complement['k'],
                                                                    sigma_2=fixed_args_mag_complement['sigma_2'],
                                                                    k_2=fixed_args_mag_complement['k_2'])
        else:
            new_Y_filtered_mag = chi2_nano_intensity_spheroid_BO_double_fixed(candidate,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,
                                                                    test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                    fixed_args_mag_complement['R_2'],
                                                                    sigma_1=fixed_args_mag_complement['sigma'],
                                                                    k_1=fixed_args_mag_complement['k'],
                                                                    sigma_2=fixed_args_mag_complement['sigma_2'],
                                                                    k_2=fixed_args_mag_complement['k_2'])
    else:
        if log_Ibg_mag is not None:
            new_Y_filtered_mag = chi2_final_intensity_spheroid_BO_single_fixed(candidate,distribution_1,formfactor_1,volume_1,test_Q,
                                                                    test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                    sigma=fixed_args_mag_complement['sigma'],
                                                                    k=fixed_args_mag_complement['k'])
        else:
            new_Y_filtered_mag = chi2_nano_intensity_spheroid_BO_single_fixed(candidate,distribution_1,formfactor_1,volume_1,test_Q,
                                                                    test_Imag,test_sigmaImag,fixed_args_mag_complement['R'],
                                                                    sigma=fixed_args_mag_complement['sigma'],
                                                                    k=fixed_args_mag_complement['k'])
    new_Y_filtered_mag = new_Y_filtered_mag.unsqueeze(-1)
    new_log_Y_filtered_mag = torch.log(new_Y_filtered_mag)
    X_filtered_mag = torch.cat([X_filtered_mag,candidate])
    Y_filtered_mag = torch.cat([Y_filtered_mag,new_Y_filtered_mag])
    log_Y_filtered_mag = torch.cat([log_Y_filtered_mag,new_log_Y_filtered_mag])
    print(f"Iteration {iteration}: best Y_filtered_mag = {Y_filtered_mag.min().item():.4f}, new candidate = {candidate}")

best_filtered_mag_idx = torch.argmin(Y_filtered_mag)
BO_filtered_mag_candidate = [X_filtered_mag[best_filtered_mag_idx].detach().cpu().numpy()]
if distribution_type == "double":
    filtered_res_mag = LM_optimize(double_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params=mag_params_filtered,kwargs=fixed_args_mag_complement,startpoints=BO_filtered_mag_candidate)
else:
    filtered_res_mag = LM_optimize(single_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params=mag_params_filtered,kwargs=fixed_args_mag_complement,startpoints=BO_filtered_mag_candidate)
print(f"Complement result mag: Parameters = {filtered_res_mag.x}, chi2_red = {filtered_res_mag.cost*2/dof_mag:.4f}")

filtered_res_mag_np = np.array(filtered_res_mag.x)
assembled_theta_nucside = assemble_theta(jointfit_params,mag_params_filtered,filtered_res_mag_np,nuc_params_filtered,jointstart_nuc_filtered,fixed_args_mag_complement)
print("Lamda = 1 (only nuclear influence) fit result:", assembled_theta_nucside)
"""
assembled_theta_nucside = [-6.82333048,-7.22396437,6.27986562,2.16129074,0.30854942,4.2834666]
assembled_theta_nucside_chi2_mag = 3.930570346645365
assembled_theta_nucside_chi2_nuc = 1.140017241955704


###### Begin Pareto Front Sweep ######
# construct starting theta

#current_theta = construct_joint_parameters(jointstart,jointstart_mag,pos_array_A_2,pos_array_mu_1,pos_array_mu_2,log_Ibg_mag,distribution_type,model_1,model_2)
#print(current_theta)
current_theta = [assembled_theta_magside]
#construct lambda checklist
list_lambda = np.concatenate((np.linspace(0.1,0.3,2,endpoint=False),np.linspace(0.3,0.7,8,endpoint=False),np.linspace(0.7,1.0,3,endpoint=False)))
print(list_lambda)

list_pareto_candidates = []
list_pareto_chi2_red_nuc = []
list_pareto_chi2_red_mag = []
list_pareto_candidates.append(assembled_theta_magside)
list_pareto_chi2_red_nuc.append(assembled_theta_magside_chi2_nuc)
list_pareto_chi2_red_mag.append(assembled_theta_magside_chi2_mag)
list_pareto_candidates.append(assembled_theta_nucside)
list_pareto_chi2_red_nuc.append(assembled_theta_nucside_chi2_nuc)
list_pareto_chi2_red_mag.append(assembled_theta_nucside_chi2_mag)

### Start Pareto Sweep ###
for weight in list_lambda:
    ### first do LM to approach solution, start sweep from lambda = 0 (magnetic scattering)
    if distribution_type == "double":
        residual_nuc_sweep = make_residuals_from_slice(double_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params,fixed_kwargs,jointfit_params)
        residual_mag_sweep = make_residuals_from_slice(double_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
    else:
        residual_nuc_sweep = make_residuals_from_slice(single_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params,fixed_kwargs,jointfit_params)
        residual_mag_sweep = make_residuals_from_slice(single_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
    joint_residual_sweep = make_weighted_joint_residuals(residual_nuc_sweep,residual_mag_sweep,weight,dof,dof_mag)
    joint_residual_sweep_nostop = make_weighted_joint_residuals(residual_nuc_sweep,residual_mag_sweep,weight,dof,dof_mag,stop=False)
    result_joint_weighted_firstfit = LM_joint_optimize(joint_residual_sweep,joint_residual_sweep_nostop,startpoints=current_theta)
    print("weight:",weight, "First approximation:", result_joint_weighted_firstfit.x)
    ### do a fine BO using this starting point
    BO_startpoint_nuc, BO_startpoint_mag = separate_nuc_mag_parameters(np.array(result_joint_weighted_firstfit.x),num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
    bounds_BO_sweep = construct_joint_bounds(BO_startpoint_nuc,BO_startpoint_mag,joint_searchwidth_fine,pos_array_A_2,pos_array_mu_1,pos_array_mu_2,log_Ibg_mag,distribution_type,model_1,model_2)
    X_sweep = bounds_BO_sweep[0]+(bounds_BO_sweep[1]-bounds_BO_sweep[0])*torch.rand(n_init_fine,bounds_BO_sweep.shape[1])
    X_sweep_nuc, X_sweep_mag = separate_nuc_mag_parameters_tensor(X_sweep,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
    if distribution_type == "double":
        Y_sweep_nuc = chi2_final_intensity_spheroid_BO_double(X_sweep_nuc,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None:
            Y_sweep_mag = chi2_final_intensity_spheroid_BO_double(X_sweep_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            Y_sweep_mag = chi2_nano_intensity_spheroid_BO_double(X_sweep_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        Y_sweep_nuc = chi2_final_intensity_spheroid_BO_single(X_sweep_nuc,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None: 
            Y_sweep_mag = chi2_final_intensity_spheroid_BO_single(X_sweep_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        else:
            Y_sweep_mag = chi2_nano_intensity_spheroid_BO_single(X_sweep_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    
    Y_sweep_nuc = Y_sweep_nuc.unsqueeze(-1)
    Y_sweep_mag = Y_sweep_mag.unsqueeze(-1)
    Y_sweep = Y_sweep_nuc*weight + Y_sweep_mag*(1-weight)
    log_Y_sweep = torch.log(Y_sweep)
    for iteration in range(max_iteration_BO_Pareto):
        model_gp = build_model_nonoise(X_sweep,log_Y_sweep,beta)
        best_f = log_Y_sweep.min().item()
        acq = LogExpectedImprovement(model_gp,best_f,maximize=False)
        candidate,_ = optimize_acqf(
        acq_function=acq,
        bounds = bounds_BO_sweep,
        q = 1,
        num_restarts=10,
        raw_samples=256,
    )
        new_X_sweep_nuc, new_X_sweep_mag = separate_nuc_mag_parameters_tensor(candidate,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
        if distribution_type == "double":
            new_Y_sweep_nuc = chi2_final_intensity_spheroid_BO_double(new_X_sweep_nuc,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
            if log_Ibg_mag is not None:
                new_Y_sweep_mag = chi2_final_intensity_spheroid_BO_double(new_X_sweep_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                new_Y_sweep_mag = chi2_nano_intensity_spheroid_BO_double(new_X_sweep_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            new_Y_sweep_nuc = chi2_final_intensity_spheroid_BO_single(new_X_sweep_nuc,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
            if log_Ibg_mag is not None: 
                new_Y_sweep_mag = chi2_final_intensity_spheroid_BO_single(new_X_sweep_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            else:
                new_Y_sweep_mag = chi2_nano_intensity_spheroid_BO_single(new_X_sweep_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        new_Y_sweep_nuc = new_Y_sweep_nuc.unsqueeze(-1)
        new_Y_sweep_mag = new_Y_sweep_mag.unsqueeze(-1)
        new_Y_sweep = new_Y_sweep_nuc*weight + new_Y_sweep_mag*(1-weight)
        new_log_Y_sweep = torch.log(new_Y_sweep)
        X_sweep = torch.cat([X_sweep,candidate])
        Y_sweep_nuc = torch.cat([Y_sweep_nuc,new_Y_sweep_nuc])
        Y_sweep_mag = torch.cat([Y_sweep_mag,new_Y_sweep_mag])
        Y_sweep = torch.cat([Y_sweep,new_Y_sweep])
        log_Y_sweep = torch.cat([log_Y_sweep,new_log_Y_sweep])
        print(f"Iteration {iteration}, Weight {weight}: best sweep target = {Y_sweep.min().item():.4f}, new candidate = {candidate}")

    best_idx_current_BO = torch.argmin(Y_sweep)
    print(f"Weight {weight}: best theta={X_sweep[best_idx_current_BO]}, best weighted chi2={Y_sweep[best_idx_current_BO]}")
    current_BO_result = [X_sweep[best_idx_current_BO].detach().cpu().numpy()]
    #rerun local optimizer
    result_joint_weighted_finalfit = LM_joint_optimize(joint_residual_sweep,joint_residual_sweep_nostop,startpoints=current_BO_result)
    #output,update for next lambda
    print("Current weight:", weight, "Best theta:", result_joint_weighted_finalfit.x)
    list_pareto_candidates.append(result_joint_weighted_finalfit.x)
    result_joint_weighted_finalfit_nuc,result_joint_weighted_finalfit_mag = separate_nuc_mag_parameters(np.array(result_joint_weighted_finalfit.x),num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
    if distribution_type == "double":
        chi2_joint_weighted_finalfit_nuc = chi2_final_intensity_spheroid_double(result_joint_weighted_finalfit_nuc,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None:
            chi2_joint_weighted_finalfit_mag = chi2_final_intensity_spheroid_double(result_joint_weighted_finalfit_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            chi2_joint_weighted_finalfit_mag = chi2_nano_intensity_spheroid_double(result_joint_weighted_finalfit_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        chi2_joint_weighted_finalfit_nuc = chi2_final_intensity_spheroid_single(result_joint_weighted_finalfit_nuc,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None:
            chi2_joint_weighted_finalfit_mag = chi2_final_intensity_spheroid_single(result_joint_weighted_finalfit_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        else:
            chi2_joint_weighted_finalfit_mag = chi2_nano_intensity_spheroid_single(result_joint_weighted_finalfit_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    list_pareto_chi2_red_nuc.append(chi2_joint_weighted_finalfit_nuc)
    list_pareto_chi2_red_mag.append(chi2_joint_weighted_finalfit_mag)
    current_theta = [np.array(result_joint_weighted_finalfit.x)]
    
#Do a reverse sweep as well
current_theta = [assembled_theta_nucside]
for weight in list_lambda:
    ### first do LM to approach solution, start sweep from lambda = 1 (nuclear scattering)
    if distribution_type == "double":
        residual_nuc_sweep = make_residuals_from_slice(double_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params,fixed_kwargs,jointfit_params)
        residual_mag_sweep = make_residuals_from_slice(double_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
    else:
        residual_nuc_sweep = make_residuals_from_slice(single_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params,fixed_kwargs,jointfit_params)
        residual_mag_sweep = make_residuals_from_slice(single_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
    joint_residual_sweep = make_weighted_joint_residuals(residual_mag_sweep,residual_nuc_sweep,weight,dof_mag,dof)#inversing order of residual input = reverse sweep
    joint_residual_sweep_nostop = make_weighted_joint_residuals(residual_mag_sweep,residual_nuc_sweep,weight,dof_mag,dof,stop=False)
    result_joint_weighted_firstfit = LM_joint_optimize(joint_residual_sweep,joint_residual_sweep_nostop,startpoints=current_theta)
    print("weight:",weight, "First approximation:", result_joint_weighted_firstfit.x)
    ### do a fine BO using this starting point
    BO_startpoint_nuc, BO_startpoint_mag = separate_nuc_mag_parameters(np.array(result_joint_weighted_firstfit.x),num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
    bounds_BO_sweep = construct_joint_bounds(BO_startpoint_nuc,BO_startpoint_mag,joint_searchwidth_fine,pos_array_A_2,pos_array_mu_1,pos_array_mu_2,log_Ibg_mag,distribution_type,model_1,model_2)
    X_sweep = bounds_BO_sweep[0]+(bounds_BO_sweep[1]-bounds_BO_sweep[0])*torch.rand(n_init_fine,bounds_BO_sweep.shape[1])
    X_sweep_nuc, X_sweep_mag = separate_nuc_mag_parameters_tensor(X_sweep,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
    if distribution_type == "double":
        Y_sweep_nuc = chi2_final_intensity_spheroid_BO_double(X_sweep_nuc,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None:
            Y_sweep_mag = chi2_final_intensity_spheroid_BO_double(X_sweep_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            Y_sweep_mag = chi2_nano_intensity_spheroid_BO_double(X_sweep_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        Y_sweep_nuc = chi2_final_intensity_spheroid_BO_single(X_sweep_nuc,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None: 
            Y_sweep_mag = chi2_final_intensity_spheroid_BO_single(X_sweep_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        else:
            Y_sweep_mag = chi2_nano_intensity_spheroid_BO_single(X_sweep_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    
    Y_sweep_nuc = Y_sweep_nuc.unsqueeze(-1)
    Y_sweep_mag = Y_sweep_mag.unsqueeze(-1)
    Y_sweep = Y_sweep_nuc*(1-weight) + Y_sweep_mag*weight
    log_Y_sweep = torch.log(Y_sweep)
    for iteration in range(max_iteration_BO_Pareto):
        model_gp = build_model_nonoise(X_sweep,log_Y_sweep,beta)
        best_f = log_Y_sweep.min().item()
        acq = LogExpectedImprovement(model_gp,best_f,maximize=False)
        candidate,_ = optimize_acqf(
        acq_function=acq,
        bounds = bounds_BO_sweep,
        q = 1,
        num_restarts=10,
        raw_samples=256,
    )
        new_X_sweep_nuc, new_X_sweep_mag = separate_nuc_mag_parameters_tensor(candidate,num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
        if distribution_type == "double":
            new_Y_sweep_nuc = chi2_final_intensity_spheroid_BO_double(new_X_sweep_nuc,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
            if log_Ibg_mag is not None:
                new_Y_sweep_mag = chi2_final_intensity_spheroid_BO_double(new_X_sweep_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
            else:
                new_Y_sweep_mag = chi2_nano_intensity_spheroid_BO_double(new_X_sweep_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            new_Y_sweep_nuc = chi2_final_intensity_spheroid_BO_single(new_X_sweep_nuc,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
            if log_Ibg_mag is not None: 
                new_Y_sweep_mag = chi2_final_intensity_spheroid_BO_single(new_X_sweep_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
            else:
                new_Y_sweep_mag = chi2_nano_intensity_spheroid_BO_single(new_X_sweep_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        new_Y_sweep_nuc = new_Y_sweep_nuc.unsqueeze(-1)
        new_Y_sweep_mag = new_Y_sweep_mag.unsqueeze(-1)
        new_Y_sweep = new_Y_sweep_nuc*(1-weight) + new_Y_sweep_mag*weight
        new_log_Y_sweep = torch.log(new_Y_sweep)
        X_sweep = torch.cat([X_sweep,candidate])
        Y_sweep_nuc = torch.cat([Y_sweep_nuc,new_Y_sweep_nuc])
        Y_sweep_mag = torch.cat([Y_sweep_mag,new_Y_sweep_mag])
        Y_sweep = torch.cat([Y_sweep,new_Y_sweep])
        log_Y_sweep = torch.cat([log_Y_sweep,new_log_Y_sweep])
        print(f"Iteration {iteration}, Weight {weight}: best sweep target = {Y_sweep.min().item():.4f}, new candidate = {candidate}")

    best_idx_current_BO = torch.argmin(Y_sweep)
    print(f"Weight {weight}: best theta={X_sweep[best_idx_current_BO]}, best weighted chi2={Y_sweep[best_idx_current_BO]}")
    current_BO_result = [X_sweep[best_idx_current_BO].detach().cpu().numpy()]
    #rerun local optimizer
    result_joint_weighted_finalfit = LM_joint_optimize(joint_residual_sweep,joint_residual_sweep_nostop,startpoints=current_BO_result)
    #output,update for next lambda
    print("Current weight:", weight, "Best theta:", result_joint_weighted_finalfit.x)
    list_pareto_candidates.append(result_joint_weighted_finalfit.x)
    result_joint_weighted_finalfit_nuc,result_joint_weighted_finalfit_mag = separate_nuc_mag_parameters(np.array(result_joint_weighted_finalfit.x),num_parameters,nuc_pos_list,mag_pos_list,log_Ibg_mag)
    if distribution_type == "double":
        chi2_joint_weighted_finalfit_nuc = chi2_final_intensity_spheroid_double(result_joint_weighted_finalfit_nuc,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None:
            chi2_joint_weighted_finalfit_mag = chi2_final_intensity_spheroid_double(result_joint_weighted_finalfit_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            chi2_joint_weighted_finalfit_mag = chi2_nano_intensity_spheroid_double(result_joint_weighted_finalfit_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        chi2_joint_weighted_finalfit_nuc = chi2_final_intensity_spheroid_single(result_joint_weighted_finalfit_nuc,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None:
            chi2_joint_weighted_finalfit_mag = chi2_final_intensity_spheroid_single(result_joint_weighted_finalfit_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        else:
            chi2_joint_weighted_finalfit_mag = chi2_nano_intensity_spheroid_single(result_joint_weighted_finalfit_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    list_pareto_chi2_red_nuc.append(chi2_joint_weighted_finalfit_nuc)
    list_pareto_chi2_red_mag.append(chi2_joint_weighted_finalfit_mag)
    current_theta = [np.array(result_joint_weighted_finalfit.x)]
###  Analyze sweep result  ###
#first discard less ideal points
keep = [True]*len(list_pareto_candidates)
for i in range(len(list_pareto_candidates)):
    for j in range(len(list_pareto_candidates)):
        if i == j:
            continue
        if (list_pareto_chi2_red_nuc[j] <= list_pareto_chi2_red_nuc[i] and list_pareto_chi2_red_mag[j] <= list_pareto_chi2_red_mag[i]) and (list_pareto_chi2_red_nuc[j] < list_pareto_chi2_red_nuc[i] or list_pareto_chi2_red_mag[j] < list_pareto_chi2_red_mag[i]):
            keep[i] = False
            break
list_pareto_front = [list_pareto_candidates[i] for i in range(len(list_pareto_candidates)) if keep[i]]
list_pareto_front_chi2_red_nuc = [list_pareto_chi2_red_nuc[i] for i in range(len(list_pareto_candidates)) if keep[i]]
list_pareto_front_chi2_red_mag = [list_pareto_chi2_red_mag[i] for i in range(len(list_pareto_candidates)) if keep[i]]
print("pareto set:", list_pareto_front)
print("chi2 red nuc values:", list_pareto_front_chi2_red_nuc)
print("chi2 red mag values:", list_pareto_front_chi2_red_mag)
#then find pareto best solution

points = sorted(zip(list_pareto_front_chi2_red_nuc,list_pareto_front,list_pareto_front_chi2_red_mag),key = lambda x: x[0])
list_pareto_front_chi2_red_nuc_sorted = [p[0] for p in points]
list_pareto_front_sorted = [p[1] for p in points]
list_pareto_front_chi2_red_mag_sorted = [p[2] for p in points]
if len(list_pareto_front_chi2_red_nuc_sorted) <= 3:
    print("Not enough pareto solutions. Returning all parameter sets.") 
else:
    x1, y1 = list_pareto_front_chi2_red_nuc_sorted[0],  list_pareto_front_chi2_red_mag_sorted[0]
    x2, y2 = list_pareto_front_chi2_red_nuc_sorted[-1], list_pareto_front_chi2_red_mag_sorted[-1]
    denom = math.hypot(x2-x1,y2-y1)
    if denom == 0:
        print("Points overlap. Check data")
    else:
        max_dist = -1.0
        best_idx = -1
        for i in range(len(list_pareto_front_chi2_red_nuc_sorted)):
            x0, y0 = list_pareto_front_chi2_red_nuc_sorted[i], list_pareto_front_chi2_red_mag_sorted[i]
            dist = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / denom
            if dist > max_dist:
                max_dist = dist
                best_idx = i

        print("Best Pareto Solution:", list_pareto_front_sorted[best_idx])
        print("Best chi2_red_nuc:", list_pareto_front_chi2_red_nuc_sorted[best_idx])
        print("Best chi2_red_mag:", list_pareto_front_chi2_red_mag_sorted[best_idx])
