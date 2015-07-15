import scipy as sp
from scipy.linalg import norm
from numpy.fft import fftfreq
import data
import copy
import math
from scipy.interpolate import interp1d
from numeric import *

def validate_sim_fun_args(input_signal, window_size=None, hop=None, max_lag=None, 
                          min_lag=None):
    """ 
        Validates the arguments window_size, hop, max_lag, min_lag of all similarity 
        functions. Returns the window size and hop in samples and adjusted min and max lags.
    """
    if window_size == None: 
        window_size = input_signal.data.size/float(input_signal.fs)
    if hop == None: 
        hop = input_signal.data.size/float(input_signal.fs)
    # Default maximum lag corresponds to 40 bpm
    if max_lag == None: 
        max_lag = 60.0/40.0
    # Default minimum lag corresponds to 250 bpm
    if min_lag == None: 
        min_lag = 60.0/250.0
    if min_lag>max_lag:
        raise ValueError("Minimum lag must be greater than maximum lag.")
    if max_lag>window_size:
        raise ValueError("window size must be greater than the maximum lag.")
    window_size_int = int(round(float(window_size)*input_signal.fs))
    hop_int = max(int(round(float(hop)*input_signal.fs)), 1)
    return window_size_int, hop_int, max_lag, min_lag
        

def corr(input_signal, window_size=None, hop=None, max_lag=None, min_lag=None, 
         normalize='one'):
    """
        Similarity computation through auto-correlation.
    """

    window_size, hop_int, max_lag, min_lag = validate_sim_fun_args(input_signal, 
                                             window_size, hop, max_lag, min_lag)
    # Segmenting
    part_sig = segmentSignal(input_signal.data, window_size, hop_int)
    result = None
    lag = sp.arange(window_size)/input_signal.fs
    ind_max = lag <= max_lag
    lag = lag[ind_max]
    ind_min = lag >= min_lag
    lag = lag[ind_min]
    for x in sp.transpose(part_sig):
        corr_result = sp.correlate(x, x, mode='full')
        corr_result = corr_result[corr_result.size/2:]
        # Normalizing so that correlation at lag=0 is 1:
        corr_result /= corr_result[0]
        corr_result = corr_result[ind_max][ind_min]
        if normalize == 'one':
            corr_result /= sp.absolute(corr_result).max()
        elif normalize == 'subband':
            corr_result *= norm(x)
        if result == None:
            result = corr_result
        else:
            result = sp.vstack((result, corr_result))
    result = result.T
    time = (sp.arange(part_sig.shape[-1])*hop_int*input_signal.period) + window_size*input_signal.period
    return data.Similarity(result, lag, time)
    

def dft(input_signal, window_size=None, hop=None, max_lag=None, min_lag=None, 
         normalize='one', zp=4):
    """
        Similarity computation through auto-correlation.
    """

    window_size, hop_int, max_lag, min_lag = validate_sim_fun_args(input_signal, 
                                         window_size, hop, max_lag, min_lag)
    # Segmenting
    part_sig = segmentSignal(input_signal.data, window_size, hop_int)
    result = None
    num_bins = zp*(2**int(math.ceil(math.log(window_size, 2))))
    freq = fftfreq(num_bins, 1.0/(input_signal.fs))
    ind_max = freq <= 1./min_lag
    freq = freq[ind_max]
    ind_min = freq >= 1./max_lag
    freq = freq[ind_min]
    win_sig = sp.signal.hamming(window_size)
    for x in sp.transpose(part_sig):
        X = sp.fftpack.fft(x*win_sig, num_bins)
        X = X[ind_max]
        X = X[ind_min]
        X = sp.absolute(X)
        if normalize == 'one':
            X /= X.max()
        elif normalize == 'subband':
            X *= norm(x)
        if result == None:
            result = X
        else:
            result = sp.vstack((result, X))
    result = result.T
    time = (sp.arange(part_sig.shape[-1]) * hop_int * input_signal.period)+window_size*input_signal.period
    return data.Similarity(result, 1./freq, time)


def sumCorr(feature, window_size=None, hop=None, max_lag=None, min_lag=None,
            normalization = None, sim_fun=corr, **fun_kw):
    """
        Add description;
        calculates and sums the similarity for each feature along time
    """
    for i in range(feature.data.shape[0]):
        if i==0:
            sim  = sim_fun(feature.getSignal(i), window_size, hop, max_lag, min_lag, 
                        normalization, **fun_kw)
        else:
            sim += sim_fun(feature.getSignal(i), window_size, hop, max_lag, min_lag, 
                        normalization, **fun_kw)
    sim.data /= feature.data.shape[0]
    return sim
    
    
def fm_acf(input_signal, window_size=None, hop=None, max_lag=None, min_lag=None, 
         normalize=None, zp=4, interp_fun='linear'):
    """"Frequency-mapped auto-correlation function."""

    window_size, hop_int, max_lag, min_lag = validate_sim_fun_args(input_signal, 
                                             window_size, hop, max_lag, min_lag)
    # Making the input signal zero-mean and with unit variance.
    aux_data = copy.copy(input_signal.data)
    aux_data -= sp.mean(aux_data)
    aux_data /= sp.std(aux_data)
    # Segmenting
    part_sig = segmentSignal(aux_data, window_size, hop_int)
    guard = 10
    uplim = int(round(float(max_lag)*input_signal.fs))+guard
    llim =  int(round(float(min_lag)*input_signal.fs))-guard
    # Calculating the spectrum:
    num_bins = zp*(2**int(math.ceil(math.log(window_size, 2))))
    # Setting window:
    win_sig = sp.signal.hamming(window_size)
    for x in sp.transpose(part_sig): # Loop over frames
        X = sp.fftpack.fft(x*win_sig, num_bins)
        freq = fftfreq(num_bins, 1.0/(input_signal.fs))
        X = sp.absolute(X[freq>=0])
        freq = freq[freq>=0]
        X /= X.max()
        # Calculating the correlation
        corrvec = sp.zeros((uplim-llim,));
        for i in range(llim, uplim):
            corrvec[i-llim] = sp.sum(x[:window_size-i]*x[i:])/(window_size-i)
        corrvec /= sp.sum(x*x) # Normalization
        # Resampling at constant units of lag
        tau = sp.arange(llim/input_signal.fs, (uplim)/input_signal.fs, 1.0/input_signal.fs)
        corrvec_fun = interp1d(tau, corrvec, kind=interp_fun)
        max_freq = 1.0/min_lag
        min_freq = 1.0/max_lag
        min_ind = sp.absolute(freq-min_freq).argmin()
        max_ind = sp.absolute(freq-max_freq).argmin()
        if freq[max_ind] > 1.0/tau[0]:
            max_ind -= 1
        if freq[min_ind] < 1.0/tau[-1]:
            min_ind += 1
        freq = freq[min_ind:max_ind]
        X = X[min_ind:max_ind]
        chosen_lag = 1.0/freq[::-1]
        corrvec_up = corrvec_fun(chosen_lag)
        corrvec_up[corrvec_up<0.0] = 0.0
        corr_result = corrvec_up*X[::-1]
        try:
            result = sp.vstack((result, corr_result))
        except NameError:
            result = corr_result
    result = result.T
    time = (sp.arange(part_sig.shape[-1])*hop_int*input_signal.period)+window_size*input_signal.period
    return data.Similarity(result, chosen_lag, time)
    
def lm_dtft(input_signal, window_size=None, hop=None, max_lag=None, min_lag=None, 
         normalize=True):
    """"
        Lag-Mapped DTFT.
        - Normalization of both auto-correlation and DFT
        - DTFT instead of DFT: allows for any delay.
    """
    # Calculating autocorrelation
    sim = corr(input_signal, window_size=window_size, hop=hop, max_lag=max_lag, 
               min_lag=min_lag, normalize=None)
    window_size, hop_int, max_lag, min_lag = validate_sim_fun_args(input_signal, 
                                             window_size, hop, max_lag, min_lag)
    # Segmenting
    part_sig = segmentSignal(input_signal.data, window_size, hop_int)
    # Obtaining desired lags
    lag = sim.lag
    discrete_freq = 2*sp.pi*input_signal.period/sim.lag
    win_sig = sp.signal.hanning(window_size)
    # Looping over each frame
    for x, acf in zip(part_sig.T, sp.atleast_2d(sim.data.T)):
        aux_sig = x * win_sig
        # han_sig = x * win_sig_han
        aux_sig -= sp.mean(aux_sig)
        x_dtft = sp.absolute(dtft(aux_sig, discrete_freq))
        if normalize:
            dtft_norm = (window_size*input_signal.period)/lag
            x_dtft /= dtft_norm
        acf /= (window_size*input_signal.period-lag)
        acf -= sp.median(acf)
        # Calculating product
        prod = x_dtft*acf
        # HWR product (removes negatives introduced by cross-correlation):
        prod[prod < 0] = 0
        # Storing result
        try:
            result = sp.vstack((result, prod))
        except NameError:
            result = prod
    return data.Similarity(result.T, sim.lag, sim.time)
    
    
def multiFeatCorr(feature, window_size=None, hop=None, max_lag=None, min_lag=None,
                  normalization = None, sim_fun=corr, **fun_kw):
    """
        Estimates the periodicity function through the correlation for several features, 
        and frames.
    """
    for i in range(feature.num_features):
        aux_sim  = sim_fun(feature.getSignal(i), window_size, hop, max_lag, min_lag, 
                        normalization, **fun_kw)
        if i == 0:
            # Initialization of the similarity data
            sim = aux_sim
            if feature.num_features != 1:
                if sim.data.ndim == 1:
                    sim.data = sp.atleast_2d(sim.data).T
        else:
            if aux_sim.data.ndim == 1:
                aux_sim.data = sp.atleast_2d(aux_sim.data).T
            sim.aggregate_feat(aux_sim)
    return sim
    
    
