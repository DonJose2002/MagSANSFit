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
if distribution_type == "log normal":
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
elif distribution_type == "double":
    if distribution_type_1 == "log normal":
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
    if distribution_type_2 == "log normal":
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
#initial guess values
n_init = 100

sigma_z_nuc = test_sigmaI/test_I
sigma_z_mag = test_sigmaImag/test_sigmaImag

avg_sigma_z_nuc = np.average(sigma_z_nuc)
avg_sigma_z_mag = np.average(sigma_z_mag)
print ("average sigma z nuc:", avg_sigma_z_nuc)
print ("average sigma z mag:", avg_sigma_z_mag)