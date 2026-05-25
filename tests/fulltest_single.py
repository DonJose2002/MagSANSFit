"""
test file with manual input for checking validity of equations, optimization methods, etc. 
Essentially a functional main.py without ui
"""
from problems.coreshell import formfactor_coreshell
from problems.distributions import distribution_normal,distribution_lognormal
from problems.ellipsoid import volume_ellipsoid,formfactor_ellipsoid
from problems.sphere import volume_sphere,formfactor_sphere
from problems.approximations import I_porod
from core.models.assembled_problem import final_intensity_spheroid,nano_intensity_spheroid,double_intensity_spheroid,single_intensity_spheroid
from core.models.BO_model import chi2_final_intensity_spheroid_BO_single,chi2_final_intensity_spheroid_BO_double,chi2_final_intensity_spheroid_BO_double_with_reward,chi2_final_intensity_spheroid_BO_single_with_reward
from core.models.loss_functions import loss_chi2,make_residual, make_residual_nostop, TargetChi2Reached
from core.utils.file_reader import file_reader_1d,file_reader_2d,file_reader_1d_nofilter
from core.acquisition.acquisition_functions import get_acq_qLogEI,get_acq_LogEI,build_model
from core.utils.helper_functions import objective,chi2_red_variance,log_chi2_red_variance
from core.optimizer.Levenberg_Marquardt import LM_optimize



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



torch.set_default_dtype(torch.double)
torch.manual_seed(0)

#BO loop settings
max_iteration_BO = 50 #maximum BO loop number
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
Log_Ibg = -6.29
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

#convert to model parameters
combinedfactor_1 = np.exp(-A_1)
sigma_rm_1 = sigma_1*Rm_1
combinedfactor_2 = np.exp(-A_2)
sigma_rm_2 = sigma_2*Rm_2
Ibg = np.exp(Log_Ibg)
C = np.exp(log_C)


#load data from txt file
test_Q, test_I, test_sigmaI, test_sigmaQ = file_reader_1d_nofilter("data\\27940.txt") #1d test case
tensor_I = torch.tensor(test_I)
tensor_sigmaI = torch.tensor(test_sigmaI)
print(tensor_sigmaI.shape)
#print(test_Q)
#log_test_I = np.log(test_I)
###computation of porod and background scattering signal

#case by case initialization
if (distribution_type == "normal") or (distribution_type == "log_normal") :
    if model_1 == "sphere":
        num_parameters_1 = 5
    elif model_1 == "ellipsoid":
        num_parameters_1 = 6
    elif model_1 == "core-shell":
        num_parameters_1 = 7
elif distribution_type == "mono":
    if model_1 == "sphere":
        num_parameters_1 = 4
    elif model_1 == "ellipsoid":
        num_parameters_1 = 5
    elif model_1 == "core-shell":
        num_parameters_1 = 6
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


model_Q = np.linspace(0.01,3,100)
#model_I = [(final_intensity_spheroid(I_porod,combinedfactor_1,distribution_lognormal,formfactor_sphere,volume_sphere,Ibg,q,Rm_1,C,sigma=sigma_rm_1)+nano_intensity_spheroid(combinedfactor_2,distribution_lognormal,formfactor_sphere,volume_sphere,q,Rm_2,sigma=sigma_rm_2)) for q in model_Q]
model_I = [final_intensity_spheroid(I_porod,combinedfactor_1,distribution_lognormal,formfactor_sphere,volume_sphere,Ibg,q,Rm_1,C,sigma=sigma_rm_1) for q in model_Q]
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
bounds_1 = torch.tensor([[max(range_Ibg[0],Log_Ibg-search_width_Ibg),min(range_Ibg[1],Log_Ibg+search_width_Ibg)],
                         [max(range_C[0],log_C-search_width_C),min(range_C[1],log_C+search_width_C)],
                         [max(range_A[0],A_1-search_width_A),min(range_A[1],A_1+search_width_A)],
                         [max(range_Rm[0],Rm_1-search_width_Rm),min(range_Rm[1],Rm_1+search_width_Rm)]])
bounds_2 = torch.tensor([[max(range_A[0],A_2-search_width_A),min(range_A[1],A_2+search_width_A)],
                         [max(range_Rm[0],Rm_2-search_width_Rm),min(range_Rm[1],Rm_2+search_width_Rm)]])

#list of parameters to optimize
params_1 = ['background_log','C_log','combinedfactor_log','R']
params_2 = ['combinedfactor_2_log','R_2']

#list of fixed parameters
fixed_kwargs = {'approx': I_porod}
# assign distribution function
distribution_1 = "mono"
distribution_2 = "mono"
if distribution_type == "log_normal":
    distribution_1 = distribution_lognormal
    bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
    params_1.append('sigma')
elif distribution_type == "normal":
    distribution_1 = distribution_normal
    bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
    params_1.append('sigma')
elif distribution_type == "Double":
    if distribution_type_1 == "log_normal":
        distribution_1 = distribution_lognormal
        bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
        params_1.append('sigma')
    elif distribution_type_1 == "normal":
        distribution_1 = distribution_normal
        bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_sigma[0],sigma_1-search_width_sigma),min(range_sigma[1],sigma_1+search_width_sigma)]])])
        params_1.append('sigma')
    if distribution_type_2 == "log_normal":
        distribution_2 = distribution_lognormal
        bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_sigma[0],sigma_2-search_width_sigma),min(range_sigma[1],sigma_2+search_width_sigma)]])])
        params_2.append('sigma_2')
    elif distribution_type_2 == "normal":
        distribution_2 = distribution_normal
        bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_sigma[0],sigma_2-search_width_sigma),min(range_sigma[1],sigma_2+search_width_sigma)]])])
        params_2.append('sigma_2')

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
    params_1.append('k')
elif model_1 == "core-shell":
    formfactor_1 = formfactor_coreshell
    volume_1 = volume_sphere
    bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_kshell[0],k_1-search_width_kshell),min(range_kshell[1],k_1+search_width_kshell)]])])
    bounds_1 = torch.cat([bounds_1,torch.tensor([[max(range_mu[0],mu_1-search_width_mu),min(range_mu[1],mu_1+search_width_mu)]])])
    params_1.append('k')
    params_1.append('mu')


if model_2 == "sphere":
    formfactor_2 = formfactor_sphere
    volume_2 = volume_sphere
elif model_2 == "ellipsoid":
    formfactor_2 = formfactor_ellipsoid
    volume_2 = volume_ellipsoid
    bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_kellipsoid[0],k_2-search_width_kellipsoid),min(range_kellipsoid[1],k_2+search_width_kellipsoid)]])])
    params_2.append('k_2')
elif model_2 == "core-shell":
    formfactor_2 = formfactor_coreshell
    volume_2 = volume_sphere
    bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_kshell[0],k_2-search_width_kshell),min(range_kshell[1],k_2+search_width_kshell)]])])
    bounds_2 = torch.cat([bounds_2,torch.tensor([[max(range_mu[0],mu_2-search_width_mu),min(range_mu[1],mu_2+search_width_mu)]])])
    params_2.append('k_2')
    params_2.append('mu_2')

fixed_kwargs['distribution'] = distribution_1
fixed_kwargs['formfactor'] = formfactor_1
fixed_kwargs['volume'] = volume_1
if distribution_type == "Double":
    fixed_kwargs['distribution_2'] = distribution_2
    fixed_kwargs['formfactor_2'] = formfactor_2
    fixed_kwargs['volume_2'] = volume_2

#assign BO bounds
if distribution_type == "Double":
    bounds = torch.cat([bounds_1,bounds_2])
    bounds = bounds.T
    params = params_1 + params_2
else:
    bounds = bounds_1.T
    params = params_1
beta = 3.0/(1/bounds.shape[1]**0.5)
dof = test_I.shape[0]-bounds.shape[1]
#initial guess values
n_init = 100
t0 = time.time()
X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, bounds.shape[1])
print(X.shape)
#print(X.shape[-1])
#print(bounds.shape)
#print(X)

if distribution_type == "Double":
    Y = chi2_final_intensity_spheroid_BO_double(X,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI)
else:
    Y = chi2_final_intensity_spheroid_BO_single(X,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI)


#if distribution_type == "Double":
#    Y,reward_variance = chi2_final_intensity_spheroid_BO_double_with_reward(X,distribution_1,distribution_2,formfactor_1,formfactor_2,volume_1,volume_2,test_Q,test_I,test_sigmaI,target_chi2,width_chi2)
#else:
#    Y,reward_variance = chi2_final_intensity_spheroid_BO_single_with_reward(X,distribution_1,formfactor_1,volume_1,test_Q,test_I,test_sigmaI,target_chi2,width_chi2)
Y = Y.unsqueeze(-1)
Y_variance = log_chi2_red_variance(Y,dof)
#reward_variance = reward_variance.unsqueeze(-1)

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
for iteration in range(max_iteration_BO):
    #with warnings.catch_warnings():
    #    warnings.simplefilter('error', category=RuntimeWarning)
    #    try: 
            #print(Y_variance.min(), Y_variance.max())
            t0 = time.time()
            model_gp = build_model(X,log_Y,Y_variance,beta)
            s2_Y = log_Y.var()
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
            new_variance = log_chi2_red_variance(new_Y,dof,s2=s2_Y)
            # Update dataset
            X = torch.cat([X, candidate])
            Y = torch.cat([Y, new_Y])
            log_Y = torch.cat([log_Y,new_log_Y])
            
            Y_variance = torch.cat([Y_variance,new_variance])
            print(f"Iteration {iteration}: best Y = {Y.min().item():.4f}, new candidate = {candidate}")
            condition = (Y >= 0.9) & (Y <= 2.0)
            count = torch.sum(condition).item()
            if count >= 3:
                indices = torch.nonzero(condition)
                print(f"Acceptable fit: chi2_red = {Y.min().item():.3f}")
                break
    #    except RuntimeWarning as e:
    #        print(f"BO loop Warning at iteration {iteration}: {e}")
best_idx = torch.argmin(Y)
print("Best theta:", X[best_idx])
print("Best chi2:", Y[best_idx])

condition = condition.squeeze(1)
indices = torch.nonzero(condition)
startpoints = X[condition]

BO_candidates = startpoints.detach().cpu().numpy()
if distribution_type == "Double":
    result = LM_optimize(double_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params=params,kwargs=fixed_kwargs,startpoints=BO_candidates)
else:
    result = LM_optimize(single_intensity_spheroid,test_Q,'Q',test_I,test_sigmaI,params=params,kwargs=fixed_kwargs,startpoints=BO_candidates)

print(f"Best result: Parameters = {result.x}, chi2_red = {result.cost*2/dof:.4f}")

J = result.jac #jacobian
H_approx = J.T @ J #hessian approximation
cov = np.linalg.inv(H_approx) #covariance matrix
uncertainties = np.sqrt(np.diag(cov)) #fit uncertainties

startpoint = torch.tensor([np.array(result.x)])
print(startpoint.shape) #1xn