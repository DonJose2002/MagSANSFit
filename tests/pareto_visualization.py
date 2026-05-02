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
st.title("Pareto Front")
###input parameters
#list_theta = [[-6.06404703, -7.18891623, 6.41047672, 2.70738054, 0.18221415, 4.37764956], 
#              [-6.82333048, -7.22396437, 6.27986562, 2.16129074, 0.30854942, 4.2834666], 
#              [-6.06546946, -7.18919135, 6.41006404, 2.70484792, 0.18303839, 4.37737027], 
#              [-6.06724513, -7.18952792, 6.40955203, 2.70172653, 0.18404745, 4.37702331], 
#              [-6.06952393, -7.18994918, 6.4088999 , 2.69778343, 0.18531164, 4.37658071], 
#              [-6.07092372, -7.19020226, 6.40850198, 2.69539484, 0.18607183, 4.37631026], 
#              [-6.07255423, -7.19049181, 6.40804094, 2.69264359, 0.18694232, 4.37599656], 
#              [-6.07447738, -7.19082635, 6.40750048, 2.68943995, 0.18794913, 4.37562836], 
#              [-6.07677939, -7.19121732, 6.4068581 , 2.68566177, 0.18912728, 4.37519007], 
#              [-6.07958378, -7.19168035, 6.40608197, 2.68113843, 0.19052496, 4.37465959], 
#              [-6.08307428, -7.19223756, 6.40512537, 2.67562357, 0.19221064, 4.37400441], 
#              [-6.08753569, -7.19292114, 6.40391704, 2.66874845, 0.19428477, 4.37317474], 
#              [-6.09343478, -7.19378007, 6.40234242, 2.65993329, 0.19690167, 4.37209025], 
#              [-6.11358539, -7.19639337, 6.39713705, 2.63182725, 0.20495282, 4.36848069], 
#              [-6.16887715, -7.2018613 , 6.38390408, 2.56591777, 0.2223941 , 4.35916986], 
#              [-6.16887718, -7.20186131, 6.38390407, 2.56591774, 0.22239411, 4.35916985],#best theta 
#              [-6.11358536, -7.19639337, 6.39713705, 2.63182727, 0.20495281, 4.36848069], 
#              [-6.09343475, -7.19378007, 6.40234243, 2.65993335, 0.19690165, 4.37209026], 
#              [-6.08753569, -7.19292114, 6.40391705, 2.66874847, 0.19428476, 4.37317475], 
#              [-6.08307427, -7.19223756, 6.40512538, 2.6756236 , 0.19221063, 4.37400441], 
#              [-6.07958378, -7.19168035, 6.40608197, 2.68113841, 0.19052497, 4.37465959], 
#              [-6.07677935, -7.19121731, 6.40685811, 2.68566183, 0.18912726, 4.37519007], 
#              [-6.07447739, -7.19082635, 6.40750048, 2.68943995, 0.18794913, 4.37562836], 
#              [-6.07255424, -7.19049181, 6.40804094, 2.69264362, 0.18694231, 4.37599657], 
#              [-6.07092373, -7.19020226, 6.40850197, 2.69539482, 0.18607184, 4.37631025], 
#              [-6.06952392, -7.18994918, 6.4088999 , 2.6977834 , 0.18531164, 4.3765807 ], 
#              [-6.06724514, -7.18952792, 6.40955203, 2.70172651, 0.18404746, 4.37702331], 
#              [-6.06546946, -7.18919135, 6.41006404, 2.70484793, 0.18303839, 4.37737027]]
list_chi2_red_nuc = [1.2798, 1.140017241955704, 1.2775962289193794, 1.274920398782468, 1.271617147259121, 1.269656975996364, 1.2674364839144396, 1.2649001583751507, 1.2619754567584829, 1.2585658724691544, 1.2545398418575695, 1.2497138700908208, 1.243823450713076, 1.2270461799487578, 1.1971093755064486, 1.197109363532205, 1.2270461880409436, 1.243823483735256, 1.2497138858268795, 1.254539860860823, 1.258565858756967, 1.2619754959852107, 1.2649001593772478, 1.2674365043972824, 1.2696569606037467, 1.271617128360783, 1.274920382501621, 1.277596230376381]
list_chi2_red_mag = [1.3487788062955082, 3.930570346645365, 1.3489010757164994, 1.349383097349714, 1.3505016141363044, 1.3514488966626523, 1.3527861382876156, 1.35466799676245, 1.357324910338095, 1.3611100902831048, 1.3665841412733357, 1.3746737378850042, 1.3869924151021806, 1.4393912760185703, 1.624945041315431, 1.624945149083599, 1.439391243649832, 1.3869923380504394, 1.3746737086608745, 1.3665841127684335, 1.3611101070424385, 1.3573248711113628, 1.354667995942553, 1.352786124632389, 1.3514489049509875, 1.350501622235595, 1.349383101419936, 1.348901075554614]
pareto_best_idx = 15
fig = go.Figure()

fig.add_trace(go.Scatter(
    x = list_chi2_red_nuc,
    y = list_chi2_red_mag,
    mode = "markers",
    name = "Pareto front",
    marker = dict(size=10,color='blue'),
))

fig.add_trace(go.Scatter(
    x = [list_chi2_red_nuc[pareto_best_idx]],
    y = [list_chi2_red_mag[pareto_best_idx]],
    mode = "markers+text",
    name = "Best Pareto Solution(knee point)",
    marker = dict(size=20,color='red'),
    text=f'({list_chi2_red_nuc[pareto_best_idx]},{list_chi2_red_mag[pareto_best_idx]})',
    textposition='top center',
    textfont=dict(size = 12),
    hovertemplate='<b>Knee point</b><br>x: %{x}<br>y: %{y}<extra></extra>'
))

fig.update_layout(
    xaxis_title = "Nuclear scattering fit chi2_red",
    yaxis_title = "Magnetic scattering fit chi2_red",
    template = "plotly_white",
    hovermode='closest'
)
st.plotly_chart(fig, use_container_width=True)