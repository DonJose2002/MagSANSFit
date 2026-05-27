# MagSANSFit
Magnetic Small Angle Neutron Scattering Fitting Workflow developed by Zilin LIU @SJTU Neutron Science Research Center

For any questions or suggestions, feel free to contact the author at jose_liu@sjtu.edu.cn

If you found this workflow useful, please consider citing the following:
(article)

## Introduction

MagSANSFit is a webapp designed for the quantitative analysis of small angle neutron scattering data under saturated magnetic field conditions, an addition to VSAS[10.1016/j.nucana.2022.100012]. It encompasses the original Bayesian Optimization method, and adapts it to the particularities of the Nuclear-Magnetic Joint Analysis problem. Instead of conventional approaches such as separating Nuclear and Magnetic scattering fitting problems, or treating it as a simple jointfit problem, this workflow treats the joint analysis problem as a coupled problem while acknowledging the possibility of disagreement between the two signals, whether the origin is physical or not, significantly increasing the reliability and physical significance of the Magnetic SANS method.

## Usage

### Installation
Make sure you create a new python environment before you start to avoid potential version conflicts.

To use this app, first clone the repository: 

```sh
git clone https://github.com/DonJose2002/MagSANSFit.git
```

Then, install the required packages:

```sh
cd MagSANSFit

pip install -r requirements.txt
```

To run the webapp, from the same directory:
```sh
streamlit run app/application.py
```


### Input
The corresponding analysis mode should be selected("Distribution Fitting" for single analysis, "Nuclear-Magnetic Joint Analysis" for joint analysis) before loading data.

MagSANSFit accepts two types of input: either .csv or .txt files. For analyzing axisymmetrical signals, there should be four columns of data: Scattering vector $Q$, in units of nm-1; Scattering intensity, in units of cm-1; the standard deviation of Scattering intensity; and the standard deviation of $Q$. For Nuclear-Magnetic Joint analysis, the aforementioned Scattering intensity should be the one measured parallel to the applied magnetic field; in addition to this, two other columns, namely the intensity perpendicular to the magnetic field and its standard deviation. Consult example.txt for more details.

### Model Selection

Currently, sphere, ellipsoids and sphere with core-shell are supported. Distribution functions include monodistribution, normal distribution and log-normal distribution. There is also the option of adding a second distribution model, not necessarily the same as the first, by selecting "Double" in "Distribution Type".

An additional radio button (in the case of Nuclear-Magnetic Joint Analysis) is provided to ignore background scattering and porod signal during magnetic scattering analysis

### Parameters

Parameter input is necessary as a first step to analysis to narrow down the search space. As a good rule of thumb, the reduced chi2 value of fit parameters should be around $10$ before attempting automatic analysis.

Intepretation:

- log_Ibg: Natural logarithm of the constant background scattering signal intensity

- log_C: Natural logarithm of the Porod signal factor, where $I_{porod} = C/Q^\gamma$

- A: The "combined factor", following the relation $exp(-A) = f_v \Delta \rho ^2 \cdot 10^7$

- Rm: Mean radius

- Sigma: relative standard deviation of the particle radius, i.e. true sigma = $\sigma \cdot R_m$

- kellipsoid: ratio between the ellipsoid minor axis (perpendicular to the axis of symmetry) and the major axis, so the ellipsoid is of the form [Rm, k\*Rm, k\*Rm]

- kshell: ratio between the core radius and the total radius

- mu: ratio between the contrast factor of the core material and the shell material

### Advanced Settings

The advanced settings table allows the user to modify variable ranges and search widths during different stages of analysis (single/joint/fine pareto sweep). You can also save your custom settings as a preset to be used in subsequent analysis.

### Analysis Configs

Similarly, options such as maximum BO loops, inital BO guesses, discrepancy analysis range are modifiable via the button Analysis Configurations.

### Launching Analysis

Simply click on the respective buttons. Output will be stored in .txt form inside the output folder.

## Interpreting Results

Single analysis follows BO-Local refinement workflow, while joint analysis goes through three stages: simple joint analysis by combining both equations, discrepancy fit where model error is introduced, and a pareto sweep. Only when the previous stage is not satisfactory will the algorithm resort to the next stage.

Therefore, interpretation of single analysis results is straightforward: simply examine the reduced chi2 value and fit uncertainties to see if these are acceptable. If not, consider trying a different model or providing a different starting point.

Supposing that the starting point of joint analysis (which is the result of two single analyses) is satisfactory by themselves, the stage at which joint analysis stops implies different physical ground truths: 

- If simple jointfit gives good results, then there is minimal disagreement between nuclear and magnetic scattering: they share the same scatterer source.

- If discrepancy is introduced, then the difference between two signals is not negligeable, yet still not great enough to raise a concern. These can be attributed to the intrinsic error of scattering models.

- If a pareto sweep is launched, then the algorithm believes that it is impossible to find a good compromise between two signals. Physically this suggests that the origins of nuclear and magnetic scattering are probably different. The algorithm then produces parameter sets on the pareto front, as well as the knee point as the "Best solution", for the user to analyse based on their knowledge of the material system in question.

