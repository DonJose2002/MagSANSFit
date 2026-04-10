import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.analytic import LogNoisyExpectedImprovement
#from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior
from gpytorch.kernels import ScaleKernel, RBFKernel
from botorch.models.transforms import Normalize,Standardize

torch.set_default_dtype(torch.double)
torch.manual_seed(0)

"""
get_acq_qLogEI: defines qLogNEI (batch log noisy expected improvement) for BO.
takes in current evaluated theta coordinates(list_x) and corresponding reduced chi^2, returns acquisition function 
"""

def get_acq_qLogEI(list_x, list_y):
    model_gp = SingleTaskGP(list_x, list_y, input_transform=Normalize(d=list_x.shape[-1]),outcome_transform=Standardize(m=1)) #create gaussian process
    mll = ExactMarginalLogLikelihood(model_gp.likelihood, model_gp)
    fit_gpytorch_mll(mll)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128])) #sampling frequency of acquisition optimization
    acq = qLogNoisyExpectedImprovement(
        model=model_gp,
        X_baseline=list_x,
        sampler=sampler,
    )
    return acq
"""
get_acq_LogEI: defines LogNEI (log noisy expected improvement) for BO.
takes in current evaluated theta coordinates(list_x) and corresponding reduced chi^2, returns acquisition function 
"""
def get_acq_LogEI(list_x,list_y,y_variance,beta):
    model_gp = SingleTaskGP(list_x, list_y, train_Yvar=y_variance,input_transform=Normalize(d=list_x.shape[-1]),outcome_transform=Standardize(m=1),
        covar_module=ScaleKernel(
            RBFKernel(
                lengthscale_prior=GammaPrior(3.0, beta)  # prevents short lengthscales
            ),
            outputscale_prior=GammaPrior(2.0, 2.0)
        ))
    mll = ExactMarginalLogLikelihood(model_gp.likelihood, model_gp)
    fit_gpytorch_mll(mll)
    acq = LogNoisyExpectedImprovement(model=model_gp,X_observed=list_x)
    return acq
"""
build_model : alternative subroutine for gp model and not acquisition function.
"""
def build_model(list_x,list_y,y_variance,beta):
    model_gp = SingleTaskGP(list_x, list_y, train_Yvar=y_variance, input_transform=Normalize(d=list_x.shape[-1]),outcome_transform=Standardize(m=1),
        covar_module=ScaleKernel(
            RBFKernel(
                ard_num_dims=list_x.shape[-1],
                lengthscale_prior=GammaPrior(3.0, beta)  # prevents short lengthscales
            ),
            outputscale_prior=GammaPrior(2.0, 2.0)
        ))
    mll = ExactMarginalLogLikelihood(model_gp.likelihood, model_gp)
    fit_gpytorch_mll(mll)
    return model_gp
