"""
test file with manual input for checking validity of equations, optimization methods, etc. 
Essentially a functional main.py without ui
"""
from problems.coreshell import formfactor_coreshell
from problems.distributions import distribution_normal,distribution_lognormal
from problems.ellipsoid import volume_ellipsoid,formfactor_ellipsoid
from problems.sphere import volume_sphere,formfactor_sphere
from problems.approximations import I_porod
from core.models.assembled_problem import final_intensity_spheroid,nano_intensity_spheroid
from core.models.loss_functions import loss_chi2
from core.utils.file_reader import file_reader_1d,file_reader_2d,file_reader_1d_nofilter


from scipy.integrate import quad
import numpy as np
import pandas as pd
import plotly.graph_objects as go

###input parameters
distribution_type = "double" #mono,normal,log normal,double
if distribution_type == "double":
    distribution_type_1 = "log normal" #mono,normal,log normal
    distribution_type_2 = "log normal" #mono,normal,log normal
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


#range settings
range_A = [0.1,10]
range_Rm = [0,20]
range_C = [-20,1]
range_sigma = [0.01,0.3]
range_kellipsoid = [0.5,3]
range_kshell = [0,1]
range_mu = [-10,10]
#convert to model parameters
combinedfactor_1 = np.exp(-A_1)
sigma_rm_1 = sigma_1*Rm_1
combinedfactor_2 = np.exp(-A_2)
sigma_rm_2 = sigma_2*Rm_2
Ibg = np.exp(Log_Ibg)
C = np.exp(log_C)


#load data from txt file
test_Q, test_I, test_sigmaI, test_sigmaQ = file_reader_1d_nofilter("data\\27940.txt") #1d test case
#log_test_I = np.log(test_I)
###computation of porod and background scattering signal

#case by case initialization
if (distribution_type == "normal") or (distribution_type == "normal") :
    if model_1 == "sphere":
        num_parameters = 5
    elif model_1 == "ellipsoid":
        num_parameters = 6
    elif model_1 == "core-shell":
        num_parameters = 7
elif distribution_type == "mono":
    if model_1 == "sphere":
        num_parameters = 4
    elif model_1 == "ellipsoid":
        num_parameters = 5
    elif model_1 == "core-shell":
        num_parameters = 6

model_Q = np.linspace(0.01,3,100)
#model_I = [(final_intensity_spheroid(I_porod,combinedfactor_1,distribution_lognormal,formfactor_sphere,volume_sphere,Ibg,q,Rm_1,C,sigma=sigma_rm_1)+nano_intensity_spheroid(combinedfactor_2,distribution_lognormal,formfactor_sphere,volume_sphere,q,Rm_2,sigma=sigma_rm_2)) for q in model_Q]
model_I = [final_intensity_spheroid(I_porod,combinedfactor_1,distribution_lognormal,formfactor_sphere,volume_sphere,Ibg,q,Rm_1,C,sigma=sigma_rm_1) for q in model_Q]
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
