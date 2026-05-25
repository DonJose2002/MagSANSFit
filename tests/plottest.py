"""
test file for visualizing results
"""
import sys
import os

sys.path.append(os.getcwd())
from problems.coreshell import formfactor_coreshell
from problems.distributions import distribution_normal,distribution_lognormal
from problems.ellipsoid import volume_ellipsoid,formfactor_ellipsoid
from problems.sphere import volume_sphere,formfactor_sphere
from problems.approximations import I_porod
from core.models.assembled_problem import final_intensity_spheroid,nano_intensity_spheroid
from core.models.loss_functions import loss_chi2
from core.models.single_loss import chi2_final_intensity_spheroid_double,chi2_final_intensity_spheroid_single,chi2_nano_intensity_spheroid_double,chi2_nano_intensity_spheroid_single
from core.utils.file_reader import file_reader_1d,file_reader_2d,file_reader_1d_nofilter


from scipy.integrate import quad
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
st.title("Toggle Nuclear/Magnetic Scattering")
###input parameters
distribution_type = "Double" #mono,normal,log normal,double
if distribution_type == "Double":
    distribution_type_1 = "log_normal" #mono,normal,log normal
    distribution_type_2 = "log_normal" #mono,normal,log normal
    distribution_weight = 0.5 #weight of the first distribution function

model_1 = "sphere" #sphere, ellipsoid, core-shell
model_2 = "sphere" #sphere, ellipsoid, core-shell

### Start Parameters
Log_Ibg = -6.07625239
log_C = -7.1911287

#Log_Ibg = -1.8736 
#log_C = -3.1409

#first distribution parameters
A_1 = 6.40700473
Rm_1 = 2.6865214
sigma_1 = 0.18886009
k_1 = 0.5
mu_1 = 0


#A_1 = 2.54
#Rm_1 = 2.009
#sigma_1 = 0.3
#k_1 = 0.5
#mu_1 = 0

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

###mag scattering start parameters
### Start Parameters
log_Ibg_mag = None
log_C_mag = None
#log_Ibg_mag = -10
#log_C_mag = -10

#Log_Ibg = -1.8736
#log_C = -3.1409

#first distribution parameters
A_1_mag = 4.37529018
Rm_1_mag = 2.6865214
sigma_1_mag = 0.18886009
k_1_mag = 0.5
mu_1_mag = 0


#A_1 = 2.54
#Rm_1 = 2.009
#sigma_1 = 0.3
#k_1 = 0.5
#mu_1 = 0

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

#convert to model parameters
combinedfactor_1_mag = np.exp(-A_1_mag)
sigma_rm_1_mag = sigma_1_mag*Rm_1_mag
combinedfactor_2_mag = np.exp(-A_2_mag)
sigma_rm_2_mag = sigma_2_mag*Rm_2_mag
#Ibg = np.exp(Log_Ibg)
#C = np.exp(log_C)

#user-related bound ranges
search_width_A = 2
search_width_Rm = 4
search_width_C = 3
search_width_Ibg = 3
search_width_sigma = 0.1
search_width_kellipsoid = 0.5
search_width_kshell = 0.3
search_width_mu = 4

#group results
theta_nuc = np.array([Log_Ibg,log_C,A_1,Rm_1,sigma_1])
theta_mag = np.array([A_1_mag,Rm_1_mag,sigma_1_mag])
#load data from txt file
test_Q, test_I, test_sigmaI, test_sigmaQ, test_Imag, test_sigmaImag = file_reader_2d("data\\27940.txt")
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
model_I_mag = [nano_intensity_spheroid(combinedfactor_1_mag,distribution_lognormal,formfactor_sphere,volume_sphere,q,Rm_1,sigma=sigma_rm_1) for q in model_Q]

#plot initial case
#model_I_chi2 = [final_intensity_spheroid(I_porod,combinedfactor_1,distribution_lognormal,formfactor_sphere,volume_sphere,Ibg,q,Rm_1,C,sigma=sigma_rm_1) for q in test_Q]
chi2_nuc = chi2_final_intensity_spheroid_single(theta_nuc,distribution_lognormal,formfactor_sphere,volume_sphere,test_Q,test_I,test_sigmaI)

chi2_mag = chi2_nano_intensity_spheroid_single(theta_mag,distribution_lognormal,formfactor_sphere,volume_sphere,test_Q,test_Imag,test_sigmaImag)
#print(chi2_mag)
col1, col2 = st.columns(2)

with col1:
    st.write("chi2_nuc:", chi2_nuc)

with col2:
    st.write("chi2_mag:", chi2_mag)
fig = go.Figure()
dataset = st.radio("Choose dataset", ["Nuclear Scattering", "Magnetic Scattering"])

fig.add_trace(go.Scatter(
    x = test_Q,
    y = np.log(test_I),
    mode = "markers",
    name = "Experimental Nuclear Scattering",
    marker = dict(size=10),
    visible=(dataset=="Nuclear Scattering")
))
fig.add_trace(go.Scatter(
    x=model_Q,
    y=np.log(model_I),
    mode="lines",
    name = "Model Nuclear Scattering",
    visible=(dataset=="Nuclear Scattering")
))
fig.add_trace(go.Scatter(
    x = test_Q,
    y = np.log(test_Imag),
    mode = "markers",
    name = "Experimental Magnetic Scattering",
    marker = dict(size=10),
    visible=(dataset=="Magnetic Scattering")
))
fig.add_trace(go.Scatter(
    x=model_Q,
    y=np.log(model_I_mag),
    mode="lines",
    name = "Model Magnetic Scattering",
    visible=(dataset=="Magnetic Scattering")
))
fig.update_layout(
    xaxis_title = "Q(nm-1)",
    yaxis_title = "log(I(Q))(cm-1)",
    template = "plotly_white"
)
st.plotly_chart(fig, width='stretch')