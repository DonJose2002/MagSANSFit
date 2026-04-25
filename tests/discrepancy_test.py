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
from core.models.BO_with_error import log_chi2_final_intensity_spheroid_BO_single,log_chi2_final_intensity_spheroid_BO_double,log_chi2_nano_intensity_spheroid_BO_double,log_chi2_nano_intensity_spheroid_BO_single
from core.models.BO_with_error_separate import log_chi2_final_intensity_spheroid_BO_double_separate,log_chi2_final_intensity_spheroid_BO_single_separate,log_chi2_nano_intensity_spheroid_BO_double_separate,log_chi2_nano_intensity_spheroid_BO_single_separate
from core.models.loss_functions import loss_chi2,make_residual, make_residual_nostop, TargetChi2Reached,make_joint_residuals,make_residuals_from_slice,noisy_loss_log_chi2,make_discrepancy_loss_from_slice,make_discrepancy_loss
from core.utils.file_reader import file_reader_1d,file_reader_2d,file_reader_1d_nofilter
from core.acquisition.acquisition_functions import get_acq_qLogEI,get_acq_LogEI,build_model,build_model_discrepancy
from core.utils.helper_functions import objective,chi2_red_variance,log_chi2_red_variance,joint_log_chi2_red_variance
from core.optimizer.Levenberg_Marquardt import LM_optimize,LM_joint_optimize
from core.optimizer.LBFGS import LBFGS_optimize



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



torch.set_default_dtype(torch.double)
torch.manual_seed(0)

#BO loop settings
max_iteration_BO = 75 #maximum BO loop number
min_discrepancy_allowance = 50 #minimum loop threshold before model starts accounting for best solutions
logspace_tolerance = 0.1 #discrepancy fit acceptance criteria
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

#discrepancy range
range_model_discrepancy = [1e-4,1]
range_model_discrepancy_log = [np.log(range_model_discrepancy[0]),np.log(range_model_discrepancy[1])]

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
test_Q, test_I, test_sigmaI, test_sigmaQ, test_Imag, test_sigmaImag = file_reader_2d("data\\1.2-100h.txt") #1d test case
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
if log_Ibg_mag is not None:
    dof_mag = dof
else:
    dof_mag = dof + 2
#initial guess values
n_init = 100


jointstart = np.array([-4.9,-6.217,5.813,8.227,0.211])
jointstart_mag = np.array([3.867,7.807,0.205])
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

print(bounds_jointfit)
#add mag specific bounds
if log_Ibg_mag is not None:
    decrement = 0
    bounds_mag_joint = [[jointstart_mag[0]-joint_searchwidth[0],jointstart_mag[0]+joint_searchwidth[0]],
                [jointstart_mag[1]-joint_searchwidth[1],jointstart_mag[1]+joint_searchwidth[1]],
                [jointstart_mag[2]-joint_searchwidth[2],jointstart_mag[2]+joint_searchwidth[2]]]
else:
    decrement = 2
    bounds_mag_joint = [[jointstart_mag[0]-joint_searchwidth[2],jointstart_mag[0]+joint_searchwidth[2]]]

if distribution_type == "double":
    bounds_mag_joint.append([jointstart_mag[pos_array_A_2-decrement]-joint_searchwidth[pos_array_A_2],jointstart_mag[pos_array_A_2-decrement]+joint_searchwidth[pos_array_A_2]])
    if model_1 == "core-shell":
        bounds_mag_joint.append([jointstart_mag[pos_array_mu_1-decrement]-joint_searchwidth[pos_array_mu_1],jointstart_mag[pos_array_mu_1-decrement]+joint_searchwidth[pos_array_mu_1]])
    if model_2 == "core-shell":
        bounds_mag_joint.append([jointstart_mag[pos_array_mu_2-decrement]-joint_searchwidth[pos_array_mu_2],jointstart_mag[pos_array_mu_2-decrement]+joint_searchwidth[pos_array_mu_2]])
else:
    if model_1 == "core-shell":
        bounds_mag_joint.append([jointstart_mag[pos_array_mu_1-decrement]-joint_searchwidth[pos_array_mu_1],jointstart_mag[pos_array_mu_1-decrement]+joint_searchwidth[pos_array_mu_1]])
print(bounds_mag_joint)
bounds_final = torch.tensor(bounds_jointfit+bounds_mag_joint)
bounds_final = bounds_final.T


###build discrepancy fit bounds from jointfit bounds
bounds_discrepancy_joint = copy.deepcopy(bounds_jointfit)
bounds_discrepancy_joint.append(range_model_discrepancy_log)
print(bounds_discrepancy_joint)
bounds_discrepancy_mag_joint = copy.deepcopy(bounds_mag_joint)
bounds_discrepancy_mag_joint.append(range_model_discrepancy_log)
print(bounds_discrepancy_mag_joint)
bounds_discrepancy_final_LBFGS = bounds_discrepancy_joint + bounds_discrepancy_mag_joint
bounds_for_LBFGS = [tuple(pair) for pair in bounds_discrepancy_final_LBFGS]
bounds_discrepancy_final = torch.tensor(bounds_discrepancy_joint+bounds_discrepancy_mag_joint)
bounds_discrepancy_final = bounds_discrepancy_final.T
print(bounds_discrepancy_final.shape)

###inital guess values
n_init_discrepancy = 100
discrepancy_X = bounds_discrepancy_final[0] + (bounds_discrepancy_final[1]-bounds_discrepancy_final[0])*torch.rand(n_init_discrepancy,bounds_discrepancy_final.shape[1])
#beta_discrepancy = 3.0/(1/bounds_discrepancy_final.shape[1]**0.5)
beta_discrepancy = 6
#construct separate starter values
discrepancy_X_nuc = discrepancy_X[:,:num_parameters+1]

discrepancy_X_mag = discrepancy_X_nuc.clone()

discrepancy_X_mag[:, nuc_pos_list] = discrepancy_X[:, num_parameters+1+torch.tensor(mag_pos_list)]
discrepancy_X_mag[:,-1] = discrepancy_X[:,-1]
if log_Ibg_mag is None:
    discrepancy_X_mag = discrepancy_X_mag[:,2:]

#initialize Y values
if distribution_type == "double":
    discrepancy_Y_nuc, discrepancy_Y_nuc_residual = log_chi2_final_intensity_spheroid_BO_double_separate(discrepancy_X_nuc,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
    if log_Ibg_mag is not None:
        discrepancy_Y_mag, discrepancy_Y_mag_residual = log_chi2_final_intensity_spheroid_BO_double_separate(discrepancy_X_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        discrepancy_Y_mag, discrepancy_Y_mag_residual = log_chi2_nano_intensity_spheroid_BO_double_separate(discrepancy_X_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
else:
    discrepancy_Y_nuc, discrepancy_Y_nuc_residual= log_chi2_final_intensity_spheroid_BO_single_separate(discrepancy_X_nuc,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
    if log_Ibg_mag is not None: 
        discrepancy_Y_mag, discrepancy_Y_mag_residual = log_chi2_final_intensity_spheroid_BO_single_separate(discrepancy_X_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    else:
        discrepancy_Y_mag, discrepancy_Y_mag_residual = log_chi2_nano_intensity_spheroid_BO_single_separate(discrepancy_X_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)

discrepancy_Y_nuc = discrepancy_Y_nuc.unsqueeze(-1)
discrepancy_Y_nuc_residual = discrepancy_Y_nuc_residual.unsqueeze(-1)
discrepancy_Y_mag = discrepancy_Y_mag.unsqueeze(-1)
discrepancy_Y_mag_residual = discrepancy_Y_mag_residual.unsqueeze(-1)
discrepancy_Y = discrepancy_Y_nuc + discrepancy_Y_mag #corresponds to tampered chi2
discrepancy_Y_residual = discrepancy_Y_nuc_residual + discrepancy_Y_mag_residual # gaussian normalization residual term
discrepancy_Y_total = discrepancy_Y + discrepancy_Y_residual
log_discrepancy_Y_total = torch.log(discrepancy_Y_total)
log_discrepancy_Y_total_best = None
acceptable_candidates = []
count = 0
print(dof,dof_mag)
#BO iteration
for iteration in range(max_iteration_BO):
    model_gp = build_model_discrepancy(discrepancy_X,log_discrepancy_Y_total,beta_discrepancy)
    best_f = discrepancy_Y_total.min().item()
    acq = LogExpectedImprovement(model_gp,best_f,maximize = False)
    candidate,_ = optimize_acqf(
        acq_function = acq,
        bounds = bounds_discrepancy_final,
        q = 1,
        num_restarts = 20,
        raw_samples = 512,
    )
    new_discrepancy_X_nuc = candidate[:,:num_parameters+1]
    new_discrepancy_X_mag = new_discrepancy_X_nuc.clone()
    new_discrepancy_X_mag[:, nuc_pos_list] = candidate[:, num_parameters+1+torch.tensor(mag_pos_list)]
    new_discrepancy_X_mag[:,-1] = candidate[:,-1]
    if log_Ibg_mag is None:
        new_discrepancy_X_mag = new_discrepancy_X_mag[:,2:]
    if distribution_type == "double":
        new_discrepancy_Y_nuc, new_discrepancy_Y_nuc_residual = log_chi2_final_intensity_spheroid_BO_double_separate(new_discrepancy_X_nuc,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None:
            new_discrepancy_Y_mag, new_discrepancy_Y_mag_residual = log_chi2_final_intensity_spheroid_BO_double_separate(new_discrepancy_X_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
        else:
            new_discrepancy_Y_mag, new_discrepancy_Y_mag_residual = log_chi2_nano_intensity_spheroid_BO_double_separate(new_discrepancy_X_mag,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_Imag,test_sigmaImag)
    else:
        new_discrepancy_Y_nuc, new_discrepancy_Y_nuc_residual = log_chi2_final_intensity_spheroid_BO_single_separate(new_discrepancy_X_nuc,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)
        if log_Ibg_mag is not None: 
            new_discrepancy_Y_mag, new_discrepancy_Y_mag_residual = log_chi2_final_intensity_spheroid_BO_single_separate(new_discrepancy_X_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
        else:
            new_discrepancy_Y_mag, new_discrepancy_Y_mag_residual = log_chi2_nano_intensity_spheroid_BO_single_separate(new_discrepancy_X_mag,distribution_1,formfactor_1,volume_1,test_Q,test_Imag,test_sigmaImag)
    new_discrepancy_Y_nuc = new_discrepancy_Y_nuc.unsqueeze(-1)
    new_discrepancy_Y_nuc_residual = new_discrepancy_Y_nuc_residual.unsqueeze(-1)
    new_discrepancy_Y_mag = new_discrepancy_Y_mag.unsqueeze(-1)
    new_discrepancy_Y_mag_residual = new_discrepancy_Y_mag_residual.unsqueeze(-1)
    new_discrepancy_Y = new_discrepancy_Y_nuc + new_discrepancy_Y_mag 
    new_discrepancy_Y_residual = new_discrepancy_Y_nuc_residual + new_discrepancy_Y_mag_residual
    new_discrepancy_Y_total = new_discrepancy_Y + new_discrepancy_Y_residual
    new_log_discrepancy_Y_total = torch.log(new_discrepancy_Y_total)
    
    discrepancy_X = torch.cat([discrepancy_X, candidate])
    discrepancy_Y_nuc = torch.cat([discrepancy_Y_nuc, new_discrepancy_Y_nuc])
    discrepancy_Y_nuc_residual = torch.cat([discrepancy_Y_nuc_residual, new_discrepancy_Y_nuc_residual])
    discrepancy_Y_mag = torch.cat([discrepancy_Y_mag, new_discrepancy_Y_mag])
    discrepancy_Y_mag_residual = torch.cat([discrepancy_Y_mag_residual, new_discrepancy_Y_mag_residual])
    discrepancy_Y = torch.cat([discrepancy_Y, new_discrepancy_Y])
    discrepancy_Y_residual = torch.cat([discrepancy_Y_residual, new_discrepancy_Y_residual])
    discrepancy_Y_total = torch.cat([discrepancy_Y_total, new_discrepancy_Y_total])
    log_discrepancy_Y_total = torch.cat([log_discrepancy_Y_total, new_log_discrepancy_Y_total])
    #jointfit_Y_variance = torch.cat([jointfit_Y_variance,new_jointfit_Y_variance])
    print(f"Iteration {iteration}: best target = {discrepancy_Y_total.min().item():.4f}, new candidate = {candidate}")
    ### check for acceptable candidates
    if log_discrepancy_Y_total_best is None:
        log_discrepancy_Y_total_best = log_discrepancy_Y_total.min().item()
    if log_discrepancy_Y_total_best > new_log_discrepancy_Y_total.item():
        log_discrepancy_Y_total_best = new_log_discrepancy_Y_total.item()
    if iteration == min_discrepancy_allowance:
        condition_firstcheck = (log_discrepancy_Y_total <= log_discrepancy_Y_total_best + logspace_tolerance).squeeze(1)# if log space chi2 is sufficiently good
        #count = torch.sum(condition_firstcheck).item()
        indices = torch.nonzero(condition_firstcheck)
        #print(indices)
        #now check equivalent chi2 and model error
        #acceptable_candidates = discrepancy_X[condition_firstcheck]
        for idx in indices:
            if (discrepancy_Y_nuc[idx].item()/dof >= 0.5) & (discrepancy_Y_nuc[idx].item()/dof <= 2) & (discrepancy_Y_mag[idx].item()/dof_mag >= 0.5) & (discrepancy_Y_mag[idx].item()/dof_mag <= 2):
                if (discrepancy_X_nuc[idx][-1].item() > 0.01) & (discrepancy_X_nuc[idx][-1].item() < 0.8) (discrepancy_X_mag[idx][-1].item() > 0.01) & (discrepancy_X_mag[idx][-1].item() < 0.8):
                    acceptable_candidates.append(discrepancy_X[idx].tolist())
                    count = count + 1
    if  iteration > min_discrepancy_allowance:
        if new_discrepancy_Y_total <= log_discrepancy_Y_total_best + logspace_tolerance:    
            if (new_discrepancy_Y_nuc.item()/dof >= 0.5) & (new_discrepancy_Y_nuc.item()/dof <= 2) & (new_discrepancy_Y_mag.item()/dof_mag >= 0.5) & (new_discrepancy_Y_mag.item()/dof_mag <= 2):
                if (new_discrepancy_X_nuc[-1].item() > 0.01) & (new_discrepancy_X_nuc[-1].item() < 0.8) (new_discrepancy_X_mag[-1].item() > 0.01) & (new_discrepancy_X_mag[-1].item() < 0.8):
                    count = count + 1
                    acceptable_candidates.append(candidate.tolist())
    if count >= 5:
        #indices = torch.nonzero(condition)
        #print(f"Acceptable fit: chi2_red = {Y.min().item():.3f}")
        break

best_idx_discrepancy = torch.argmin(discrepancy_Y_total)
print("Best theta:", discrepancy_X[best_idx_discrepancy])
print("Best combined chi2:", discrepancy_Y[best_idx_discrepancy])
print("acceptable candidates:", acceptable_candidates)



if not acceptable_candidates:
    startpoints_discrepancyfit = [discrepancy_X[best_idx_discrepancy].detach().cpu().numpy()]
else:
    startpoints_discrepancyfit = np.array(acceptable_candidates)



###L-BFGS-B 
LBFGS_params = params + ['discrepancy_nuc'] + params_mag_1 + ['discrepancy_mag']
if distribution_type == "double":
    loss_nuc = make_discrepancy_loss_from_slice(double_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params,fixed_kwargs,LBFGS_params,'discrepancy_nuc')
    loss_mag = make_discrepancy_loss_from_slice(double_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,LBFGS_params,'discrepancy_mag')
    combined_loss = make_discrepancy_loss(loss_nuc,loss_mag)
    result_LBFGS = LBFGS_optimize(combined_loss,startpoints_discrepancyfit,bounds_for_LBFGS)
else:
    loss_nuc = make_discrepancy_loss_from_slice(single_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params,fixed_kwargs,LBFGS_params,'discrepancy_nuc')
    loss_mag = make_discrepancy_loss_from_slice(single_intensity_spheroid_mag,test_Q,'Q',test_Imag,test_sigmaImag,params_mag,fixed_kwargs,LBFGS_params,'discrepancy_mag')
    combined_loss = make_discrepancy_loss(loss_nuc,loss_mag)
    result_LBFGS = LBFGS_optimize(combined_loss,startpoints_discrepancyfit,bounds_for_LBFGS)
### post treatment
for points in result_LBFGS:
    print(points)
    m = len(points)
    points_nuc = points[:num_parameters+1]
    points_mag = copy.deepcopy(points_nuc)
    for j, src_col in zip(nuc_pos_list, mag_pos_list):
        points_mag[j] = points[num_parameters+1+src_col]
    points_mag[-1] = points[-1]
    if log_Ibg_mag is None:
        points_mag = points_mag[2:]
    equivalent_chi2_nuc = loss_nuc(points)
    model_discrepancy_nuc = np.exp(points_nuc[-1])
    equivalent_chi2_mag = loss_mag(points)
    model_discrepancy_mag = np.exp(points_mag[-1])
    print("Nuclear discrepancy fit result:")
    print("chi2_eq:", equivalent_chi2_nuc)
    print("Model discrepancy:", model_discrepancy_nuc)
    print("Magnetic discrepancy fit result:")
    print("chi2_eq:", equivalent_chi2_mag)
    print("Model discrepancy:", model_discrepancy_mag)
    