""" This module contains scripts for audio feature extraction."""

import exceptions
import warnings
import scipy as sp
import scipy.fftpack as fft
import matplotlib.mlab as ml
from data import *
import numeric
import copy

def spectrogram(input_signal, window_length=20e-3, hop=10e-3, 
                windowing_function=sp.hanning, dft_length = None, zp_flag = False):
    """ Calculates the Short-Time Fourier Transform a signal.
    
    Given an input signal, it calculates the DFT of frames of the signal and stores them
    in bi-dimensional Scipy array. 
        
    **Args**:        
        - window_len (float):length of the window in seconds (must be positive).
        - window (callable): a callable object that receives the window length in samples 
          and returns a numpy array containing the windowing function samples.
        - hop (float): frame hop between adjacent frames in seconds.
        - final_time (positive integer): time (in seconds) up to which the spectrogram is 
          calculated.
        - zp_flag (bool): a flag indicating if the *Zero-Phase Windowing* should be 
          performed.
        
    **Returns**:
        A :class:`spectrogram` object containing the result of the STFT calculation in its
         ``data`` attribute.
        
    """
    # Converting window_length and hop from seconds to number of samples:
    window_length = int(round(window_length*input_signal.fs))
    feat_fs  = 1.0/hop # New sampling rate of the feature data equals the hop
    hop = max(int(round(hop*input_signal.fs)),1)
    spec, time, frequency = numeric.STFT(input_signal.data, window_length, hop, 
                                         windowing_function, dft_length, zp_flag)
    # Converting indices to seconds and Hz:
    time /= input_signal.fs
    frequency *= input_signal.fs
    # Returning new features:
    return Feature(data=spec, time_index=time, feature_index=frequency, 
                   fs=feat_fs, name="spectrogram", feat_axis_label = "Frequency (Hz)")
                   
def respectrogram(input_signal, window_length=20e-3, hop=10e-3, 
                windowing_function=sp.hanning, dft_length = None, zp_flag = False):
    """ Calculates the Short-Time Fourier Transform a signal.
    
    Given an input signal, it calculates the DFT of frames of the signal and stores them
    in bi-dimensional Scipy array. 
        
    **Args**:        
        - window_len (float):length of the window in seconds (must be positive).
        - window (callable): a callable object that receives the window length in samples 
          and returns a numpy array containing the windowing function samples.
        - hop (float): frame hop between adjacent frames in seconds.
        - final_time (positive integer): time (in seconds) up to which the spectrogram is 
          calculated.
        - zp_flag (bool): a flag indicating if the *Zero-Phase Windowing* should be 
          performed.
        
    **Returns**:
        A :class:`spectrogram` object containing the result of the STFT calculation in its
         ``data`` attribute.
        
    """
    # Converting window_length and hop from seconds to number of samples:
    fs = input_signal.fs
    window_length = int(round(window_length*fs))
    feat_fs  = 1.0/hop # New sampling rate of the feature data equals the hop
    hop = max(int(round(hop*fs)),1)
    spec, time, frequency = numeric.RESTFT(input_signal.data, fs, window_length, hop, 
                                         windowing_function, dft_length)
    # Converting indices to seconds and Hz:
    time /= fs
    frequency *= fs
    # Returning new features:
    return Feature(data=spec, time_index=time, feature_index=frequency, 
                   fs=feat_fs, name="spectrogram", feat_axis_label = "Frequency (Hz)")
                   
                   

def melScaleSpectrogram(input, nfilts = 40, minfreq = 20, maxfreq = None, **kw):
    """This function converts a Spectrogram with linearly spaced frequency components 
    to the Mel scale.
    
        Given an input signal, it calculates the DFT of frames of the signal and stores 
        them in bi-dimensional Scipy array. 
    
    
    **Args**:        
        - window_len (float): length of the window in seconds (must be positive).
        - window (callable): a callable object that receives the window length in samples 
          and returns a numpy array containing the windowing function samples.
        - hop (float): frame hop between adjacent frames in seconds.
        - final_time (positive integer): time (in seconds) up to which the spectrogram is
          calculated.
        - zp_flag (bool): a flag indicating if the *Zero-Phase Windowing* should be performed.
    **Returns**:
        A :class:`Spectrogram` object containing the result of the STFT calculation in 
        its ``data`` attribute.
        
    **Raises**:
    """
    if isinstance(input, Signal):
        if isinstance(input, Feature):
            # Assuming input is a feature with frequency (in Hz) indices...
            spec = input
        else:
            # Assuming the input is another specialization of signal:
            val_kw, remaining_kw = numeric.getValidKeywords(kw, spectrogram)
            spec = spectrogram(input, **val_kw)
    else:
        raise TypeError('This class only accepts instances of Signal as input.')
    if maxfreq==None:
        maxfreq=spec.feature_index.max()
    (wts, frequency) = numeric.fft2mel(spec.feature_index, spec.fs, nfilts, minfreq, maxfreq)
    new_data = sp.dot(wts, sp.sqrt(sp.absolute(spec.data)**2))
    return Feature(data=new_data, time_index=spec.time_index,
                       feature_index=frequency, fs=spec.fs, signal=spec.signal,
                       name="mel-spectrogram",  feat_axis_label = "Frequency (Hz)")

def logMelSpectrogram(input, **kw):
    """ Calculates the Short-Time Fourier Transform a signal.
    
    Given an input signal, it calculates the DFT of frames of the signal and stores them
    in bi-dimensional Scipy array. 
    
    **Args**:        
        - window_len (float): length of the window in seconds (must be positive).
        - window (callable):  a callable object that receives the window length in samples 
          and returns a numpy array containing the windowing function samples.
        - hop (float): frame hop between adjacent frames in seconds.
        - final_time (positive integer): time (in seconds) up to which the spectrogram is 
          calculated.

    **Returns**:
        A :class:`Spectrogram` object containing the result of the STFT calculation in its
         ``data`` attribute.
        
    **Raises**:
                   
    """
    if isinstance(input, Signal):
        if input.name=='mel-spectrogram':
            # Only needs to take the log:
            output = copy.copy(input)  
        else:
            # Map spectrogram from linear to mel scale:
            output = melScaleSpectrogram(input, **kw)
    else:
        raise TypeError("Input must be an instance of Signal.")
    # Taking log of the output (with care to avoid log(0))
    # Forcing log(0)=-300 instead of -Inf (avoids numerical problemas)
    output.data[output.data==0.0] = sp.exp(-200)
    output.data = sp.log(output.data)
    output.name = "log-"+output.name
    return output   

def calculateDelta(input, delta_filter_length=3):
    """ This function calculates the delta coefficients of a given feature.
   
    **Args**:        
        - input: feature object
        - delta_filter_length (int): length of the filter used to calculate the Delta 
          coefficients. Must be an odd number.
    **Returns**:
        - A :class:`Feature` object containing the result of the calculation.      
    """
    output = copy.copy(input)
    output.name = "delta-"+output.name
    output.data = numeric.deltas(output.data, delta_filter_length)
    return output


def sumFeatures(input):
    """ This function sums all features along a given frame. A weighting function
    can be provided in order to stress some features.
   
    **Args**:        
        - input: feature object
    **Returns**:
        - A :class:`Feature` object containing the result of the calculation.      
    """
    output = copy.copy(input)
    output.name = "sum-"+output.name
    output.data = sp.sum(output.data, axis=0)
    return output
    
    
def calculateEnergy(input):
    """ This function calculates the energy of each feature, i.e. x'x. 
   
    **Args**:        
        - input: feature object
    **Returns**:
        - A :class:`Feature` object containing the result of the calculation.      
    """
    output = copy.copy(input)
    output.name = "energy-"+output.name
    output.data = sp.absolute(output.data)**2
    return output
       
    
def halfWaveRectification(input):
    """ Half-wave rectifies features.
    
        All feature values below zero are assigned to zero.
   
    **Args**:        
        - input: feature object
        - delta_filter_length (int): length of the filter used to calculate the Delta 
          coefficients. Must be an odd number.
    **Returns**:
        A :class:`Feature` object containing the result of the calculation.
        
    **Raises**:
        - ValueError when the input features are complex.
    """
    output = copy.copy(input)
    output.data = input.data.copy()
    output.name = "hwr-"+output.name
    if output.data.dtype!=complex:
        output.data[output.data<0] = 0.0
    else:
        raise ValueError('Cannot half-wave rectify a complex signal.')
    return output
    
def spectralFlux(input, sum=True, log=False, mel=False, downsample=None, **kw):
    """ Spectral Flux-like features.
    
        This performs the following calculations to the input signal:
        
        input->(Downsampling)->STFT->(Mel scale)->(Log)->Diff->HWR->(Sum)
        
        Parenthesis denote optional steps.
       
    **Args**:        
        - input: signal object.
        - sum (bool): true if the features are to be summed for each frame.
        - log (bool): true if the features energy are to be converted to dB.
        - mel (bool): true if the features are to be mapped in the Mel scale.
        - downsample (int): downsampling factor, if None no downsampling is performed.
        - **kw: these keyword arguments are passed down to each of functions used to
        obtain the feature.
    **Returns**:
        A :class:`Feature` object containing the result of the calculation.
        
    **Raises**:
        - 
    """

    signal = copy.copy(input)
    # Downsampling
    if downsample!=None:
        signal.data = sp.signal.decimate(signal.data, downsample, n=100, ftype='fir')
    # STFT
    val_kw, remaining_kw = numeric.getValidKeywords(kw, spectrogram)
    feat = spectrogram(signal, **val_kw)
    # Mel scale mapping
    if mel:
        val_kw, remaining_kw = numeric.getValidKeywords(remaining_kw, melScaleSpectrogram)
        feat = melScaleSpectrogram(feat, **val_kw)
    else:
        # Taking the absolute value of the features
        feat.data = sp.absolute(feat.data)
    # Log
    if log:
        feat.data[feat.data==0.0] = sp.exp(-200)
        feat.data = 20*sp.log10(feat.data)
        feat.name = "log-"+feat.name
    # Diff and Half-wave rectification
    feat = calculateDelta(feat, delta_filter_length=1)
    feat = halfWaveRectification(feat)
    # Sum
    if sum:
        feat = sumFeatures(feat)
    # Adapting name:
    feat.name = 'Spectral Flux: '+feat.name   
    # Return
    return feat


def spectralFluxFromSpec(input, sum=True, log=False, mel=False, **kw):
    """ This function calculates the spectral flux from a spectrogram."""
    # Mel scale mapping
    if mel:
        val_kw, remaining_kw = numeric.getValidKeywords(kw, melScaleSpectrogram)
        feat = melScaleSpectrogram(input, **val_kw)
    else:
        # Taking the absolute value of the features
        feat = copy.copy(input)
        feat.data = sp.absolute(input.data)
    # Log
    if log:
        feat.data[feat.data==0.0] = sp.exp(-200)
        feat.data = 20*sp.log10(feat.data)
        feat.name = "log-"+feat.name
    # Diff and Half-wave rectification
    feat = calculateDelta(feat, delta_filter_length=1)
    feat = halfWaveRectification(feat)
    # Sum
    if sum:
        feat = sumFeatures(feat)
    # Adapting name:
    feat.name = 'Spectral Flux: '+feat.name   
    # Return
    return feat


def percMedianDecomposition(input, feat_per=spectralFluxFromSpec, kw_spec={}, kw_per={}):
    """This function decomposes a spectrogram into a percussive and a harmonic part."""
#     import matplotlib.pyplot as plt
    feat = spectrogram(input, **kw_spec)
    plt.figure(1)
    feat.data = sp.absolute(feat.data)
#     ind_min = sp.absolute(feat.feature_index-20.0).argmin()
#     ind_max = sp.absolute(feat.feature_index-10000.0).argmin()
#     plt.imshow((20*sp.log10(feat.data[ind_min:ind_max, :])), origin = 'lower', 
#                   aspect = 'auto', interpolation='nearest')
    # Calculating filter length
    har_len = round(0.5/feat.period)
    per_len = round(300/(feat.feature_index[1]-feat.feature_index[0]))
#     har_len = 17
#     per_len = 17
    if (har_len % 2)==0: har_len += 1
    if (per_len % 2)==0: per_len += 1
    # Obtaining decomposed spectrum
    S_har_data, S_per_data = numeric.median_decomposition(feat.data, hor_filter_len=har_len, ver_filter_len=per_len)
    S_per = copy.copy(feat)
#     S_har = copy.copy(feat)
#     S_har.data = S_har_data
    S_per.data = S_per_data 
    S_per.name = 'Percurssive-'+S_per.name
#     S_har.name = 'Harmonic-'+S_har.name
#     min_freq = 20
#     max_freq = 50000
#     plt.figure(2)
#     plt.imshow((20*sp.log10(S_per_data[ind_min:ind_max, :])), origin = 'lower', 
#                       aspect = 'auto', interpolation='nearest')
#     plt.figure(3)
#     plt.imshow((20*sp.log10(S_har_data[ind_min:ind_max, :])), origin = 'lower', 
#                       aspect = 'auto', interpolation='nearest')
#     max_energy_bin = sp.sum(feat.data).argmax()
#     plt.plot(feat.time_index, feat.data[max_energy_bin, :], 'r--')
#     plt.plot(S_har.time_index, S_har.data[max_energy_bin, :], 'g')
#     plt.plot(S_per.time_index, S_per.data[max_energy_bin, :], 'b')
#     plt.legend(['SpecFlux', 'Harmonic', 'Percurssive'])
#     plt.show()
#     transform = lambda x: 20*sp.log10(x)
#     plt.figure(1)
#     feat.plot(transform=transform)
#     plt.title('Spec')
#     plt.ylim(min_freq, max_freq)
#     plt.figure(2)
#     S_per.plot(transform=transform)
#     plt.title('Harmonic part')
#     plt.ylim(min_freq, max_freq)    
#     plt.figure(3)
#     S_har.plot(transform=transform)
#     plt.title('Percurssive part')
#     plt.ylim(min_freq, max_freq)
#     plt.show()   
    feat = feat_per(S_per, **kw_per)
#    feat_har = feat_har(S_har, **kw_har)
#     plt.figure(4)
#     plt.plot(feat.time_index, feat.data)
#     plt.show()
#     plt.figure(5)
#     plt.plot(feat_har.data)
#     plt.show()
#     return [feat_per, feat_har]
# input.data = S_per
# lS = logMelSpectrogram(input)
# dS = calculateDelta(lS)
# h = halfWaveRectification(dS)
# f = sumFeatures(h)
#     input.getSignal(10).plot()
#     plt.figure(2)
#     input.data = S_per_data
#     input.plot()
#     plt.figure(3)
#     input.data = S_har_data
#     input.plot
    return feat
    
    
def ssePHDecomp(input, feat_per=spectralFluxFromSpec, per_len=17, har_len=17,
                kw_spec={}, kw_per={}, kw_har={}):
    """This function decomposes a spectrogram into a percussive and a harmonic part."""
    feat = spectrogram(input, **kw_spec)
    feat.data = sp.absolute(feat.data)
    # Calculating filter length
    # Obtaining decomposed spectrum
    S_per_data, S_har_data = numeric.sse_decomposition(feat.data, hor_filter_len=per_len, ver_filter_len=har_len)
    feat.data = S_per_data
    feat.name = 'Percurssive-'+feat.name
    feat = feat_per(feat, **kw_per)
    feat_har = copy.copy(feat)
    feat_har.data = S_har_data
    feat_har = feat_per(feat_har, **kw_har)
    return feat, feat_har

if __name__ == "__main__":
    pass