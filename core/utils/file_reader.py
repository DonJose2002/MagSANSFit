import pandas as pd
import numpy as np
def file_reader_1d(path): #reads 1d scattering signal file and returns numpy array
    df = pd.read_csv(path, sep=r'\s+', header=0)
    filtered_df = df[df.iloc[:,1]>0] #filter negative values
    filtered_arr = filtered_df.to_numpy()
    Q = filtered_arr[:,0]
    I = filtered_arr[:,1]
    sigmaI = filtered_arr[:,2]
    sigmaQ = filtered_arr[:,3]
    return Q,I,sigmaI,sigmaQ

def file_reader_1d_nofilter(path): #reads 1d scattering signal file and returns numpy array, nofilter version for already treated data
    df = pd.read_csv(path, sep=r'\s+', header=0)
    filtered_arr = df.to_numpy()
    Q = filtered_arr[:,0]
    I = filtered_arr[:,1]
    sigmaI = filtered_arr[:,2]
    sigmaQ = filtered_arr[:,3]
    return Q,I,sigmaI,sigmaQ

def file_reader_2d(path): #reads scattering signal file and returns numpy array
    df = pd.read_csv(path, sep=r'\s+', header=0)
    filtered_df = df[(df.iloc[:,1]>0) & (df.iloc[:,4]>df.iloc[:,1])] #filter negative and abnormal magnetic scattering signals
    filtered_arr = filtered_df.to_numpy()
    Q = filtered_arr[:,0]
    I_nuc = filtered_arr[:,1]
    sigmaI_nuc = filtered_arr[:,2]
    sigmaQ_nuc = filtered_arr[:,3]
    I_mag = filtered_arr[:,4] - filtered_arr[:,1]
    sigmaI_mag = filtered_arr[:,2] + filtered_arr[:,5]
    return Q,I_nuc,sigmaI_nuc,sigmaQ_nuc,I_mag,sigmaI_mag
"""
def data_interpreter_1d(data_arr): #separates 1d signal info into corresponding arrays
    #first filter all negative signals
    mask = data_arr[:,1] > 0
    filtered_arr = data_arr[mask]
    Q = filtered_arr[:,0]
    I = filtered_arr[:,1]
    sigmaI = filtered_arr[:,2]
    sigmaQ = filtered_arr[:,3]
    return Q,I,sigmaI,sigmaQ

def data_interpreter_2d(data_arr): #separates d signal info into corresponding arrays
    #first filter all negative signals
    mask = data_arr[:,1] > 0
    filtered_arr = data_arr[mask]
    Q = filtered_arr[:,0]
    I = filtered_arr[:,1]
    sigmaI = filtered_arr[:,2]
    sigmaQ = filtered_arr[:,3]
    return Q,I,sigmaI,sigmaQ
"""
