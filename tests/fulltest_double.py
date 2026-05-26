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
from core.models.loss_functions import loss_chi2,make_residual, make_residual_nostop, TargetChi2Reached,make_joint_residuals,make_residuals_from_slice
from core.utils.file_reader import file_reader_1d,file_reader_2d,file_reader_1d_nofilter
from core.acquisition.acquisition_functions import get_acq_qLogEI,get_acq_LogEI,build_model
from core.utils.helper_functions import objective,chi2_red_variance,log_chi2_red_variance,joint_log_chi2_red_variance
from core.optimizer.Levenberg_Marquardt import LM_optimize,LM_joint_optimize


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
from botorch.acquisition.analytic import LogNoisyExpectedImprovement
import warnings
import time
import copy



torch.set_default_dtype(torch.double)
torch.manual_seed(0)

#BO loop settings
max_iteration_BO = 75 #maximum BO loop number
#reduced chi2 transform options
target_chi2 = 1
width_chi2 = 0.3

###input parameters
distribution_type = "log_normal" #mono,normal,log normal,double
if distribution_type == "Double":
    distribution_type_1 = "log_normal" #mono,normal,log normal
    distribution_type_2 = "log_normal" #mono,normal,log normal
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

#user-related bound ranges
search_width_A_jointfit = 1
search_width_Rm_jointfit = 1
search_width_C_jointfit = 1
search_width_Ibg_jointfit = 1
search_width_sigma_jointfit = 0.1
search_width_kellipsoid_jointfit = 0.5
search_width_kshell_jointfit = 0.3
search_width_mu_jointfit = 1

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
if (distribution_type == "normal") or (distribution_type == "log_normal") :
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
elif distribution_type == "Double":
    if (distribution_type_1 == "normal") or (distribution_type_1 == "log_normal") :
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
    if (distribution_type_2 == "normal") or (distribution_type_2 == "log_normal") :
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
bounds_2 = torch.tensor([[max(range_A[0],A_2-search_width_A),min(range_A[1],A_2+search_width_A)],
                         [max(range_Rm[0],Rm_2-search_width_Rm),min(range_Rm[1],Rm_2+search_width_Rm)]])
bounds_box_2 = [range_A,range_Rm]
joint_searchwidth_2 = [search_width_A_jointfit,search_width_Rm_jointfit]
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
#list of fixed parameters
fixed_kwargs = {'approx': I_porod}
# assign distribution function
distribution_1 = "mono"
distribution_2 = "mono"
if distribution_type == "log_normal":
    distribution_1 = distribution_lognormal
    bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
    bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
    params_1.append('sigma')
    common_params.append('sigma')
    bounds_box_1.append(range_sigma)
    joint_searchwidth_1.append(search_width_sigma_jointfit)
elif distribution_type == "normal":
    distribution_1 = distribution_normal
    bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
    bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
    params_1.append('sigma')
    common_params.append('sigma')
    bounds_box_1.append(range_sigma)
    joint_searchwidth_1.append(search_width_sigma_jointfit)
elif distribution_type == "Double":
    if distribution_type_1 == "log_normal":
        distribution_1 = distribution_lognormal
        bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
        bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
        params_1.append('sigma')
        common_params.append('sigma')
        bounds_box_1.append(range_sigma)
        joint_searchwidth_1.append(search_width_sigma_jointfit)
    elif distribution_type_1 == "normal":
        distribution_1 = distribution_normal
        bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
        bounds_1_mag = torch.cat([bounds_1_mag,torch.tensor([[max(range_sigma[0],sigma_1_mag-search_width_sigma),min(range_sigma[1],sigma_1_mag+search_width_sigma)]])])
        params_1.append('sigma')
        common_params.append('sigma')
        bounds_box_1.append(range_sigma)
        joint_searchwidth_1.append(search_width_sigma_jointfit)
    if distribution_type_2 == "log_normal":
        distribution_2 = distribution_lognormal
        bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_sigma[0],sigma_2-search_width_sigma),min(range_sigma[1],sigma_2+search_width_sigma)]])])
        bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_sigma[0],sigma_2_mag-search_width_sigma),min(range_sigma[1],sigma_2_mag+search_width_sigma)]])])
        params_2.append('sigma_2')
        common_params.append('sigma_2')
        bounds_box_2.append(range_sigma)
        joint_searchwidth_2.append(search_width_sigma_jointfit)
    elif distribution_type_2 == "normal":
        distribution_2 = distribution_normal
        bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_sigma[0],sigma_2-search_width_sigma),min(range_sigma[1],sigma_2+search_width_sigma)]])])
        bounds_2_mag = torch.cat([bounds_2_mag,torch.tensor([[max(range_sigma[0],sigma_2_mag-search_width_sigma),min(range_sigma[1],sigma_2_mag+search_width_sigma)]])])
        params_2.append('sigma_2')
        common_params.append('sigma_2')
        bounds_box_2.append(range_sigma)
        joint_searchwidth_2.append(search_width_sigma_jointfit)

fixed_kwargs['distribution'] = distribution_1
if distribution_type == "Double":
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
    params_1.append('mu')
    bounds_box_1.append(range_mu)
    joint_searchwidth_1.append(search_width_mu_jointfit)


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
    params_2.append('mu_2')
    bounds_box_2.append(range_mu)
    joint_searchwidth_2.append(search_width_mu_jointfit)

fixed_kwargs['distribution'] = distribution_1
fixed_kwargs['formfactor'] = formfactor_1
fixed_kwargs['volume'] = volume_1
if distribution_type == "Double":
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
if distribution_type == "Double":
    bounds = torch.cat([bounds_1,bounds_2])
    bounds_mag = torch.cat([bounds_1_mag,bounds_2_mag])
    bounds = bounds.T
    bounds_mag = bounds_mag.T
    params = params_1 + params_2
    bounds_box_np_1 = np.array(bounds_box_1)
    joint_searchwidth = joint_searchwidth_1 + joint_searchwidth_2
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
    if model_1 == "core-shell":
        pos_array_mu_1 = bounds_box_np.shape[0]-1
        nuc_pos_list.append(pos_array_mu_1)
        mag_pos_list.append(mag_pos_list[-1]+1)
        params_mag_1.append('mu_mag')

params_mag = params_mag_1 + common_params
jointfit_params = params + params_mag_1
#beta = 3.0/(1/bounds.shape[1]**0.5)

beta = 6 #smaller beta makes search more aggressive
dof = test_I.shape[0]-bounds.shape[1] #degree of freedom for chi2
if log_Ibg_mag is not None:
    dof_mag = dof
else:
    dof_mag = dof + 2
#initial guess values
n_init = 100

t0 = time.time()
X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, bounds.shape[1])
X_mag = bounds_mag[0] + (bounds_mag[1]-bounds_mag[0])*torch.rand(n_init, bounds_mag.shape[1])
#print(X.shape)
#print(X.shape[-1])
#print(bounds.shape)
#print(X)

if distribution_type == "Double":
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
   


#if distribution_type == "Double":
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
            if distribution_type == "Double":
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
if distribution_type == "Double":
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
            Y_mag_variance = log_chi2_red_variance(Y_mag,dof_mag,s2=s2_Y_mag)
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
            if distribution_type == "Double":
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
if distribution_type == "Double":
    result_mag = LM_optimize(double_intensity_spheroid,test_Q,'Q',test_Imag,test_sigmaImag,params_firstfit_mag,fixed_kwargs,BO_candidates_mag)
else:
    result_mag = LM_optimize(single_intensity_spheroid,test_Q,'Q',test_Imag,test_sigmaImag,params_firstfit_mag,fixed_kwargs,BO_candidates_mag)

print(f"Best result: Parameters = {result_mag.x}, chi2_red = {result_mag.cost*2/dof_mag:.4f}")

J_mag = result_mag.jac #jacobian
H_approx_mag = J_mag.T @ J_mag #hessian approximation
cov_mag = np.linalg.inv(H_approx_mag) #covariance matrix
uncertainties_mag = np.sqrt(np.diag(cov_mag)) #fit uncertainties




### Joint fit ###
jointstart = np.array(result.x)
jointstart_mag = np.array(result_mag.x)

#jointstart = np.array([-6.82333048,-7.22396437,6.27986562,2.16129074,0.30854942])
#jointstart_mag = np.array([4.37764956,2.70738054,0.18221415])
#construct BO bounds
bounds_jointfit = [[jointstart[0]-joint_searchwidth[0],jointstart[0]+joint_searchwidth[0]],
                   [jointstart[1]-joint_searchwidth[1],jointstart[1]+joint_searchwidth[1]],
                   [jointstart[2]-joint_searchwidth[2],jointstart[2]+joint_searchwidth[2]]]


for i in range(3, jointstart.size):
    if i != pos_array_A_2 and i != pos_array_mu_1 and i != pos_array_mu_2:
        if log_Ibg_mag is not None:
            i_mag = i
        else:
            i_mag = i-2 #decrement by 2 to match position
        current_bound = [max(jointstart[i]-joint_searchwidth[i],jointstart_mag[i_mag]-joint_searchwidth[i]),min(jointstart[i]+joint_searchwidth[i],jointstart_mag[i_mag]+joint_searchwidth[i])]
    else:
        current_bound = [jointstart[i]-joint_searchwidth[i],jointstart[i]+joint_searchwidth[i]]
    
    bounds_jointfit.append(current_bound)

#add mag specific bounds
if log_Ibg_mag is not None:
    decrement = 0
    bounds_mag_joint = [[jointstart_mag[0]-joint_searchwidth[0],jointstart_mag[0]+joint_searchwidth[0]],
                [jointstart_mag[1]-joint_searchwidth[1],jointstart_mag[1]+joint_searchwidth[1]],
                [jointstart_mag[2]-joint_searchwidth[2],jointstart_mag[2]+joint_searchwidth[2]]]
else:
    decrement = 2
    bounds_mag_joint = [[jointstart_mag[0]-joint_searchwidth[2],jointstart_mag[0]+joint_searchwidth[2]]]

if distribution_type == "Double":
    bounds_mag_joint.append([jointstart_mag[pos_array_A_2-decrement]-joint_searchwidth[pos_array_A_2],jointstart_mag[pos_array_A_2-decrement]+joint_searchwidth[pos_array_A_2]])
    if model_1 == "core-shell":
        bounds_mag_joint.append([jointstart_mag[pos_array_mu_1-decrement]-joint_searchwidth[pos_array_mu_1],jointstart_mag[pos_array_mu_1-decrement]+joint_searchwidth[pos_array_mu_1]])
    if model_2 == "core-shell":
        bounds_mag_joint.append([jointstart_mag[pos_array_mu_2-decrement]-joint_searchwidth[pos_array_mu_2],jointstart_mag[pos_array_mu_2-decrement]+joint_searchwidth[pos_array_mu_2]])
else:
    if model_1 == "core-shell":
        bounds_mag_joint.append([jointstart_mag[pos_array_mu_1-decrement]-joint_searchwidth[pos_array_mu_1],jointstart_mag[pos_array_mu_1-decrement]+joint_searchwidth[pos_array_mu_1]])

bounds_final = torch.tensor(bounds_jointfit+bounds_mag_joint)
bounds_final = bounds_final.T

###inital guess values
n_init_jointfit = 100
jointfit_X = bounds_final[0] + (bounds_final[1]-bounds_final[0])*torch.rand(n_init_jointfit,bounds_final.shape[1])
#beta_jointfit = 3.0/(1/bounds_final.shape[1]**0.5)
beta_jointfit = 6
#construct separate starter values
jointfit_X_nuc = jointfit_X[:,:num_parameters]
jointfit_X_mag = jointfit_X_nuc.clone()

jointfit_X_mag[:, nuc_pos_list] = jointfit_X[:, num_parameters+torch.tensor(mag_pos_list)]
if log_Ibg_mag is None:
    jointfit_X_mag = jointfit_X_mag[:,2:]
#initialize Y values
if distribution_type == "Double":
    jointfit_Y_nuc = chi2_final_intensity_spheroid_BO_double(jointfit_X_nuc,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
    if log_Ibg_mag is not None:
        jointfit_Y_mag = chi2_final_intensity_spheroid_BO_double(jointfit_X_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        jointfit_Y_mag = chi2_nano_intensity_spheroid_BO_double(jointfit_X_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
else:
    jointfit_Y_nuc = chi2_final_intensity_spheroid_BO_single(jointfit_X_nuc,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
    if log_Ibg_mag is not None: 
        jointfit_Y_mag = chi2_final_intensity_spheroid_BO_single(jointfit_X_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    else:
        jointfit_Y_mag = chi2_nano_intensity_spheroid_BO_single(jointfit_X_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)

jointfit_Y_nuc = jointfit_Y_nuc.unsqueeze(-1)
jointfit_Y_mag = jointfit_Y_mag.unsqueeze(-1)
jointfit_Y = jointfit_Y_nuc + jointfit_Y_mag 



log_Y_jointfit = torch.log(jointfit_Y)

#BO iteration
for iteration in range(max_iteration_BO):
    s2_jointfit = log_Y_jointfit.var()
    jointfit_Y_variance = joint_log_chi2_red_variance(jointfit_Y_nuc,jointfit_Y_mag,dof,dof_mag,s2_jointfit)
    model_gp = build_model(jointfit_X,log_Y_jointfit,jointfit_Y_variance,beta_jointfit)

    acq = LogNoisyExpectedImprovement(model_gp,jointfit_X,maximize = False)
    candidate,_ = optimize_acqf(
        acq_function = acq,
        bounds = bounds_final,
        q = 1,
        num_restarts = 20,
        raw_samples = 512,
    )
    new_jointfit_X_nuc = candidate[:,:num_parameters]
    new_jointfit_X_mag = new_jointfit_X_nuc.clone()
    new_jointfit_X_mag[:, nuc_pos_list] = candidate[:, num_parameters+torch.tensor(mag_pos_list)]
    if log_Ibg_mag is None:
        new_jointfit_X_mag = new_jointfit_X_mag[:,2:]
    if distribution_type == "Double":
        new_jointfit_Y_nuc = chi2_final_intensity_spheroid_BO_double(new_jointfit_X_nuc,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None:
            new_jointfit_Y_mag = chi2_final_intensity_spheroid_BO_double(new_jointfit_X_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            new_jointfit_Y_mag = chi2_nano_intensity_spheroid_BO_double(new_jointfit_X_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        new_jointfit_Y_nuc = chi2_final_intensity_spheroid_BO_single(new_jointfit_X_nuc,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None: 
            new_jointfit_Y_mag = chi2_final_intensity_spheroid_BO_single(new_jointfit_X_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        else:
            new_jointfit_Y_mag = chi2_nano_intensity_spheroid_BO_single(new_jointfit_X_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    new_jointfit_Y_nuc = new_jointfit_Y_nuc.unsqueeze(-1)
    new_jointfit_Y_mag = new_jointfit_Y_mag.unsqueeze(-1)
    new_jointfit_Y = new_jointfit_Y_nuc + new_jointfit_Y_mag
    new_log_Y_jointfit = torch.log(new_jointfit_Y)
    #new_jointfit_Y_variance = joint_log_chi2_red_variance(new_jointfit_Y_nuc,new_jointfit_Y_mag,dof)
    jointfit_X = torch.cat([jointfit_X, candidate])
    jointfit_Y_nuc = torch.cat([jointfit_Y_nuc, new_jointfit_Y_nuc])
    jointfit_Y_mag = torch.cat([jointfit_Y_mag, new_jointfit_Y_mag])
    jointfit_Y = torch.cat([jointfit_Y, new_jointfit_Y])
    log_Y_jointfit = torch.cat([log_Y_jointfit, new_log_Y_jointfit])
    #jointfit_Y_variance = torch.cat([jointfit_Y_variance,new_jointfit_Y_variance])
    print(f"Iteration {iteration}: best target = {jointfit_Y.min().item():.4f}, new candidate = {candidate}")

    condition_jointfit = (jointfit_Y >= 1.6) & (jointfit_Y <= 4.0)
    count = torch.sum(condition_jointfit).item()
    if count >= 5:
        #indices = torch.nonzero(condition)
        #print(f"Acceptable fit: chi2_red = {Y.min().item():.3f}")
        break

best_idx_jointfit = torch.argmin(jointfit_Y)
print("Best theta:", jointfit_X[best_idx_jointfit])
print("Best combined chi2:", jointfit_Y[best_idx_jointfit])
if torch.any(condition_jointfit).item():
    condition_jointfit = condition_jointfit.squeeze(1)
    #indices_jointfit = torch.nonzero(condition_jointfit)
    startpoints_jointfit = jointfit_X[condition_jointfit]
    if count != 1:
        BO_candidates_jointfit = startpoints_jointfit.detach().cpu().numpy()
    else:
        BO_candidates_jointfit = [jointfit_X[best_idx_jointfit].detach().cpu().numpy()]
else:
    BO_candidates_jointfit = [jointfit_X[best_idx_jointfit].detach().cpu().numpy()]
    #BO_candidates_jointfit.reshape(1,bounds_final.shape[1])


if distribution_type == "Double":
    residual_nuc = make_residuals_from_slice(double_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params,fixed_kwargs,jointfit_params)
    residual_mag = make_residuals_from_slice(double_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
    joint_residual_nostop = make_joint_residuals(residual_nuc,residual_mag)
    joint_residual = make_joint_residuals(residual_nuc,residual_mag,dof=dof,dof_2=dof_mag)
    result_jointfit_1 = LM_joint_optimize(joint_residual,joint_residual_nostop,startpoints=BO_candidates_jointfit)
else:
    residual_nuc = make_residuals_from_slice(single_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params,fixed_kwargs,jointfit_params)
    residual_mag = make_residuals_from_slice(single_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,jointfit_params)
    joint_residual_nostop = make_joint_residuals(residual_nuc,residual_mag)
    joint_residual = make_joint_residuals(residual_nuc,residual_mag,dof=dof,dof_2=dof_mag)
    result_jointfit_1 = LM_joint_optimize(joint_residual,joint_residual_nostop,startpoints=BO_candidates_jointfit)

J_jointfit_1 = result_jointfit_1.jac #jacobian
H_approx_jointfit_1 = J_jointfit_1.T @ J_jointfit_1 #hessian approximation
cov_jointfit_1 = np.linalg.inv(H_approx_jointfit_1) #covariance matrix
uncertainties_jointfit_1 = np.sqrt(np.diag(cov_jointfit_1)) #fit uncertainties

print(f"Best result: Parameters = {result_jointfit_1.x}, chi2_red = {result_jointfit_1.cost*2/dof:.4f}")

#best_res = [-6.07625239,-7.1911287,6.40700473,2.6865214,0.18886009,4.37529018]
