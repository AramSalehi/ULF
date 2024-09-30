import numpy as np
from scipy import constants

# Functions of the sequences
def spin_echo_seq(TR, TE, t1, t2, m0):
    # Spin echo formula
    TR_div = np.divide(TR, t1, out=np.zeros_like(t1), where=t1!=0)
    TE_div = np.divide(TE, t2, out=np.zeros_like(t2), where=t2!=0)
    return np.abs(m0 * (1 - np.exp(-TR_div)) * np.exp(-TE_div))   

def Gradient_seq(TR, TE, t1, t2_star, m0, alp):
    # Gradient echo formula
    TR_div = np.divide(-TR, t1, out=np.zeros_like(t1), where=t1!=0)
    TE_div = np.divide(-TE, t2_star, out=np.zeros_like(t2_star), where=t2_star!=0)
    a = np.sin(alp) * (1 - np.exp(TR_div)) * np.exp(TE_div)
    b = 1 - np.cos(alp) * np.exp(TR_div)
    im = np.divide(a, b, out=np.zeros_like(a), where=a!=0)
    return np.abs( m0 * im )

def IN_seq(TR, TE, TI, t1, t2, m0):
    # Inversion recovery formula
    a = 2 * np.exp(-np.divide(TI,t1, out=np.zeros_like(t1), where=t1!=0))
    b = np.exp(-np.divide(TR, t1, out=np.zeros_like(t1), where=t1!=0))
    c = 1-a+b
    return np.abs(m0 * c * np.exp(-np.divide(TE,t2, out=np.zeros_like(t2), where=t2!=0)))

def DoubleInversion_seq(TR, TE, TI1, TI2, t1, t2, m0):
    # Double inversion recovery formula
    E1 = np.exp(-np.divide(TI1, t1, out=np.zeros_like(t1), where=t1!=0))
    E2 = np.exp(-np.divide(TI2, t1, out=np.zeros_like(t1), where=t1!=0))
    Ec = np.exp(-np.divide(TR, t1, out=np.zeros_like(t1), where=t1!=0))
    Etau = np.exp(np.divide((TE/2), t1, out=np.zeros_like(t1), where=t1!=0))
    Ee = np.exp(-np.divide(TE,t2, out=np.zeros_like(t2), where=t2!=0))
    
    Mz = 1 - 2*E2 + 2*E1*E2 - Ec*(2*Etau - 1)
    return np.abs(m0 * Mz * Ee)

def SSFP_Echo_seq(TR, t1, t2, m0, alp):
    # SSFP FID formula
    E1 = np.divide(-TR, t1, out=np.zeros_like(t1), where=t1!=0)
    E2 = np.divide(-TR, t2, out=np.zeros_like(t2), where=t2!=0)
    a = 1 - E2**2
    b = (1 - E1*np.cos(alp))**2 - (E1 - np.cos(alp)) * (E2**2)    
    r = np.sqrt(np.abs(np.divide(a,b,out=np.zeros_like(b), where=b!=0)))
    return  np.abs(m0 * np.tan(alp/2) * (1 - (1 - E1*np.cos(alp)) * r))

def Diffusion_seq(TR, TE, t1, t2, m0, b, D):
    # Diffusion formula with predefine b
    TR_div = np.divide(TR, t1, out=np.zeros_like(t1), where=t1!=0)
    TE_div = np.divide(TE, t2, out=np.zeros_like(t2), where=t2!=0)
    spin = np.abs(m0 * (1 - np.exp(-TR_div)) * np.exp(-TE_div))
    return np.abs(spin * np.exp(-b*D))    

def T2_ETL_decay(c,t2,TE,ETL):
    # function computing the T2 decay signal  with respect to the ETL
    # Use broadcasting to speed up computation
    
    t = TE * ETL    
    time = np.linspace(0,t,c)[:, None]
    sig = np.zeros(c)
    
    t2_flat = t2.flatten()[None]
    ma = t2_flat[t2_flat != 0]
    
    T2decay = np.exp((-time)/ma)
    sig = np.sum(T2decay, 1)

    return sig

def TSE_seq(TR, TE, ETL, m0, t1, t2, t2dec, c):
    # Turbo spin echo formula
    spin = spin_echo_seq(TR, TE, t1, t2, m0) # Computing a spin echo
    sig = T2_ETL_decay(c,t2dec,TE,ETL) # signal of T2* decay
    FT = np.fft.fftshift(np.fft.fft(sig))    # FT of T2star decay   
    im = np.zeros((spin.shape))
    
    if len(im.shape) == 2:
        for i in range(spin.shape[1]):           # Convolution of each row with the lorentzian
            im[:,i] = np.convolve(FT.real, spin[:,i], mode='same')
            
    elif len(im.shape) == 3:
        for j in range(spin.shape[2]):
            for i in range(spin.shape[1]):
                im[:,i,j] = np.convolve(FT.real, spin[:,i,j], mode='same')

    return np.abs(im)


