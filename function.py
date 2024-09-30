import numpy as np
import cv2
from Sequences import *

def create_3D_noisy_and_clean_data(FOV, Resolution, Bandwidth, seq, TR, TE, TI, TI2, alpha, noise_factor\
                                   ,T1_3D, T2_3D, M0_3D, B1_3D, flipangle_3D, t2_star_3D, ADC_3D):
    """
    Function that will create two 3D simulation of low-field MRI sequences, one without noise and one with noise
    
    Inputs:
    
    //////// parameters to be chosen for runing the sequence ////////
    FOV          -> 1x3 array of the field of view
    Resolution   -> 1x3 array of the resolution
    Bandwidth    -> Bandwidth of aquisition
    seq          -> string defining the sequence to simulate; {'SE','GE','IN','Double IN','FLAIR','Dif'} 
    The seq strings correspond to these sequences; {spin echo, gradient echo, inversion recovery, double inversion recovery, FLAIR, diffusion}
    TR           -> repetition time, must be in milli-seconds (ex: 3000 for 3 seconds, 50 for 50 milli-seconds)
    TE           -> echo time, must be in milli-seconds (ex: 160 for 0.160 seconds, 50 for 50 milli-seconds)
    TI           -> inversion time, must be in milli-seconds (ex: 650 for 0.650 seconds, 50 for 50 milli-seconds)
    TI2          -> second inversion time, must be in milli-seconds (ex: 150 for 0.150 seconds, 50 for 50 milli-seconds)
    alpha        -> flip angle, between {0-90}
    noise_factor -> multiplying noise factor

    //////// low-field systems and physiological maps ////////
    T1_3D        -> T1 relaxation map
    T2_3D        -> T2 relaxation map
    M0_3D        -> Proton density map
    B1_3D        -> B1 map
    flipangle_3D -> flipangle tensor map
    t2_star_3D   -> t2 star tensor map
    ADC_3D       -> apparent diffusion coefficient (ADC) tensor map
    
    Outputs:
    clean_data -> the simulated tensor sequence without noise
    noisy_data -> the simulated tensor sequence with noise
    
    """
    
    # Final data matrix size (field of view / reolution) 
    Data_mat = np.divide(FOV, Resolution)
    Data_mat = [int(Data_mat[0]), int(Data_mat[1]), int(Data_mat[2])]
    
    # Divide by 1000 to have the values in seconds
    TR = np.divide(TR,1000) 
    TE = np.divide(TE,1000)
    TI = np.divide(TI,1000)
    TI2 = np.divide(TI2,1000)

    ##### NOISE #####
    # The noise is a 3D tensor that will be added to the data
    const = 1.0410172321868288e-07   # From brain images of low field
    var = const * Bandwidth          # Equation of the variance of noise to add (variance of noise is proportional/corelated to Bandwidth * const)
    std = np.sqrt(var)
    noise = np.random.normal(0,np.abs(std),Data_mat)

    # constant that multiply the amount of noise to add
    noise = noise*noise_factor
    
    # Computing the 3D sequence and resizing
    if seq == 'SE':
        Data_3D = spin_echo_seq(TR, TE, T1_3D, T2_3D, M0_3D)
    elif seq == 'GE':
        angle = flipangle_3D/alpha
        Data_3D =  Gradient_seq(TR, TE, T1_3D, t2_star_3D, M0_3D, angle);
    elif seq == 'IN':
        Data_3D = IN_seq(TR, TE, TI, T1_3D, T2_3D, M0_3D)
    elif seq == 'Double IN':
        Data_3D = DoubleInversion_seq(TR, TE, TI, TI2, T1_3D, T2_3D, M0_3D)
    elif seq == 'FLAIR':
        TI = np.log(2) * 3.695 
        Data_3D = IN_seq(TR, TE, TI, T1_3D, T2_3D, M0_3D)
    elif seq == 'Dif':
        Data_3D = Diffusion_seq(TR, TE, T1_3D, T2_3D, M0_3D, b, ADC_3D)

    # Multiplying the data by B1 map and normalizing  
    Data_3D = np.multiply(Data_3D, B1_3D)
    Data_3D = Data_3D/np.max(Data_3D)

    # Resizing the data
    n_seq = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2]));  clean_data = np.zeros((Data_mat))
    for x in range(T1_3D.shape[0]):
        n_seq[x,:,:] = cv2.resize(Data_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
    for x in range(Data_mat[1]):
        clean_data[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))

    # Adding the noise
    noisy_data = clean_data + noise

    # print('Simulated sequence: ' + seq)
    # print('Shape of the noisy_data tensor: ' + str(noisy_data.shape))
    # print('Shape of the clean_data tensor: ' + str(clean_data.shape))
    
    return clean_data, noisy_data