""" 
    This is module contains several auxiliary functions for Tempo Analysis and Beat tracking in Python.
    
    All auxiliary functions operate directly over array-like objects (not specific class
    of the RA package). 

 """
import scipy as sp
import scipy.fftpack as fft
import scipy.signal
import exceptions
import warnings
import matplotlib.pyplot as plt

def STFT(x, window_length, hop, windowing_function = sp.hanning, dft_length = None, 
         zp_flag = False):
    """ Calculates the Short-Time Fourier Transform a signal.
    
    Given an input signal, it calculates the DFT of frames of the signal and stores them
    in bi-dimensional Scipy array. 
        
    **Args**:        
        window_len (float): length of the window in seconds (must be positive).
        window (callable): a callable object that receives the window length in samples 
                           and returns a numpy array containing the windowing function 
                           samples.
        hop (float): frame hop between adjacent frames in seconds.
        final_time (positive integer): time (in seconds) up to which the spectrogram is 
                                       calculated.
        zp_flag (bool): a flag indicating if the *Zero-Phase Windowing* should be 
                        performed.
        
    **Returns**:
        A :class:`Spectrogram` object containing the result of the STFT calculation in its ``data`` attribute.
        
    **Raises**:
                   
    """
    # Checking input:
    if(x.ndim != 1):
        raise AttributeError("Data must be one-dimensional.")
    # Window length must be odd:
    if window_length%2 == 0: 
        window_length = window_length + 1
    # DFT length is equal the window_len+1 (always even)
    if dft_length == None:
        dft_length = window_length + 1
    # If dft_length was set by the user, it should always be larger than the window length.
    if dft_length < window_length: 
        warnings.warn("DFT length is smaller than window length.", RuntimeWarning)
    # Partitioning the input signal:
    part_sig = segmentSignal(x, window_length, hop)
    no_cols = part_sig.shape[1]
    # Applying the window:
    window = windowing_function(window_length)
    win_sig = part_sig * sp.transpose(sp.tile(window, (no_cols, 1)))
    # Zero-phase windowing:
    if zp_flag: 
        win_sig = fft.fftshift(win_sig, axes = 0)
    # Taking the FFT of the partitioned signal
    spec = fft.fftshift(fft.fft(win_sig, n = dft_length, axis = 0), axes = 0)
    # Normalizing energy
    spec /= sp.sum(window)
    # Calculating time and frequency indices for the data
    frequency = fft.fftshift(fft.fftfreq(dft_length))
    time = sp.arange(no_cols)*float(hop) + ((window_length-1)/2)
    # Creating output spectrogram
    return spec, time, frequency
    
    
def RESTFT(x, Fs, window_length, hop, windowing_function = sp.hanning, dft_length = None):
    """ Calculates the Short-Time Fourier Transform a signal.
    
    Given an input signal, it calculates the DFT of frames of the signal and stores them
    in bi-dimensional Scipy array. 
        
    **Args**:        
        window_len (float): length of the window in seconds (must be positive).
        window (callable): a callable object that receives the window length in samples 
                           and returns a numpy array containing the windowing function 
                           samples.
        hop (float): frame hop between adjacent frames in seconds.
        final_time (positive integer): time (in seconds) up to which the spectrogram is 
                                       calculated.
        zp_flag (bool): a flag indicating if the *Zero-Phase Windowing* should be 
                        performed.
        
    **Returns**:
        A :class:`Spectrogram` object containing the result of the STFT calculation in its ``data`` attribute.
        
    **Raises**:
                   
    """
    # Checking input:
    if(x.ndim != 1):
        raise AttributeError("Data must be one-dimensional.")
    # Window length must be odd:
    if window_length%2 == 0: 
        window_length = window_length + 1
    # DFT length is equal the window_len+1 (always even)
    if dft_length == None:
        dft_length = window_length + 1
    # If dft_length was set by the user, it should always be larger than the window length.
    if dft_length < window_length: 
        warnings.warn("DFT length is smaller than window length.", RuntimeWarning)
    # Partitioning the input signal:
    part_sig = segmentSignal(x, window_length, hop)
    no_cols = part_sig.shape[1]
    # Applying the window (STFT calculation):
    window = windowing_function(window_length)
    # Time-ramped window:
    t_ramp = sp.arange(-(window_length-1)/2, 1+(window_length-1)/2)
    t_window = (t_ramp/Fs)*window
    # Frequency-ramp window:
    f_ramp = sp.arange(-1.0, 0.99999, 2.0/window_length)
    f_window = -sp.imag(fft.ifft(fft.fft(window)*fft.fftshift(f_ramp)))
    # Applying time ramped window
    win_sig = part_sig * sp.transpose(sp.tile(window, (no_cols, 1)))
    t_win_sig = part_sig * sp.transpose(sp.tile(t_window, (no_cols, 1)))
    f_win_sig = part_sig * sp.transpose(sp.tile(f_window, (no_cols, 1)))
    # Taking the FFT of the partitioned signal
    spec = fft.fftshift(fft.fft(win_sig, n=dft_length, axis=0), axes=0)
    t_spec = fft.fftshift(fft.fft(t_win_sig, n=dft_length, axis=0), axes=0)
    f_spec = fft.fftshift(fft.fft(f_win_sig, n=dft_length, axis=0), axes=0)
    # Power STFT
    pspec = sp.absolute(spec)**2
    # Calculating time and frequency indices for the data
    frequency = fft.fftshift(fft.fftfreq(dft_length))
    time = sp.arange(no_cols)*float(hop) + ((window_length-1)/2)
    # Finding (discrete) offsets
    f_off = sp.int_((-sp.imag((f_spec*sp.conj(spec)))/pspec)*dft_length)
    t_off = sp.int_(sp.real((t_spec*sp.conj(spec)))/pspec)
    respec = sp.ones(pspec.shape)*(10.**(-200/20))
    # Reassignment
    for ind_line in range(dft_length):
        for ind_col in range(no_cols):
            safe_line = sp.minimum(sp.maximum(ind_line+f_off[ind_line, ind_col], 0), dft_length-1)
            safe_col = sp.minimum(sp.maximum(ind_col+t_off[ind_line, ind_col], 0), no_cols-1)
            respec[safe_line, safe_col] += pspec[ind_line, ind_col]
    return respec, time, frequency
 
 
def segmentSignal(signal, window_len, hop):
    """ Segmentation of an array-like input:
    
    Given an array-like, this function calculates the DFT of frames of the signal and stores them
    in bi-dimensional Scipy array. 
    
    
    **Args**:
        signal (array-like): object to be windowed. Must be a one-dimensional array-like object.     
        window_len (int): window size in samples.
        hop (int): frame hop between adjacent frames in seconds.
        
    **Returns**:
        A 2-D numpy array containing the windowed signal. Each element of this array X
        can be defined as:
        
        X[m,n] = x[n+Hm]
        
        where, H is the HOP in samples, 0<=n<=N, N = window_len, and 0<m<floor(((len(x)-N)/H)+1).
        
    **Raises**:
        AttributeError if signal is not one-dimensional.
        ValueError if window_len or hop  are not strictly positives.
    """
    if(window_len<=0 or hop<=0):
        raise ValueError("window_len and hop values must be strictly positive numbers.")
    if(signal.ndim!=1):
        raise AttributeError("Input signal must be one dimensional.")
    # Calculating the number of columns:
    no_cols = sp.floor((sp.size(signal)-window_len)/float(hop))+1
    # Windowing indices (which element goes to which position in the windowed matrix).
    ind_col = sp.tile(sp.arange(window_len, dtype=long), (no_cols,1))
    ind_line = sp.tile(sp.arange(no_cols, dtype=long)*hop, (window_len, 1))
    ind = sp.transpose(ind_col)+ind_line
    # Partitioned signal:
    part_sig = signal[ind].copy()
    # Windowing partitioned signal
    return part_sig
    
    
def peakPicking(input_array):
    """ Segmentation and Windowing of an array-like input:
    
    Given an array-like, this function calculates the DFT of frames of the signal and stores them
    in bi-dimensional Scipy array. 
    
    
    **Args**:
        signal (array-like): object to be windowed. Must be a one-dimensional array-like object.     
        window_len (int): window size in samples.
        hop (int): frame hop between adjacent frames in seconds.
        
    **Returns**:
        A 2-D numpy array containing the windowed signal. Each element of this array X
        can be defined as:
        
        X[m,n] = x[n+Hm]
        
        where, H is the HOP in samples, 0<=n<=N, N = window_len, and 0<m<floor(((len(x)-N)/H)+1).
        
    **Raises**:
        AttributeError if signal is not one-dimensional.
        ValueError if window_len or hop  are not strictly positives.
    """
    "Detects all peaks in a numpy array. The indices of the array are returned."
    aux  = scipy.signal.lfilter([1, -1], 1, input_array, axis = 0)
    aux = sp.sign(aux)
    aux = scipy.signal.lfilter([1, -1], 1, aux, axis = 0)
    ind = aux==-2
    ind = sp.insert(ind, ind.shape[0]-1, False, axis=0)
    ind = sp.delete(ind, 0, axis=0)
    return ind


def fft2mel(freq, fs, nfilts, minfreq, maxfreq):
    """ This method returns a 2-D Numpy array of weights that map a linearly spaced spectrogram
    to the Mel scale.
    
    **Args**:        
        freq (1-D Numpy array): frequency of the components of the DFT.
        fs (float): sampling rate of the signal.
        nfilts (): number of output bands.
        width (): 
        minfreq (): frequency of the first MEL coefficient.
        maxfreq (): frequency of the last MEL coefficient.
        
    **Returns**:
        A :class:numpay.ndarray that when multiplied with the spectrogram converts it to the mel scale.
        The center frequencies in Hz of the Mel bands.

    **Raises**:  
        """
    minmel = hz2mel(minfreq)
    maxmel = hz2mel(maxfreq)  
    binfrqs = mel2hz(minmel+sp.arange(nfilts+2)/(float(nfilts)+1)*(maxmel-minmel))
    wts = sp.zeros((nfilts, (freq.size)))
    for i in range(nfilts):
        slp = binfrqs[i + sp.arange(3)]
        loslope = (freq - slp[0])/(slp[1] - slp[0])
        hislope = (slp[2] - freq)/(slp[2] - slp[1])
        wts[i,:] = sp.maximum(0.0, sp.minimum(loslope, hislope));
    wts[:, freq < 0] = 0
    wts = sp.dot(sp.diag(2./(binfrqs[2+sp.arange(nfilts)]-binfrqs[sp.arange(nfilts)])), wts);
    binfrqs = binfrqs[1:nfilts+1]
    return wts, binfrqs
   
   
def hz2mel(f_hz):
    """ Converts a given frequency in Hz to the Mel scale.
    
    **Args**: 
        f_hz (Numpy array): Array containing the frequencies in HZ that should be converted.
        
    **Returns**:
        A Numpy array (of same shape as f_zh) containing the converted frequencies.
    
    """
    f_0 = 0 
    f_sp = 200.0/3.0 # Log step
    brkfrq = 1000.0 # Frequency above which the distribution stays linear.
    brkpt  = (brkfrq - f_0)/f_sp # First Mel value for linear region.
    logstep = sp.exp(sp.log(6.4)/27) # Step in the log region
    z_mel = sp.where(f_hz < brkfrq, (f_hz - f_0)/f_sp, brkpt + (sp.log(f_hz/brkfrq))/sp.log(logstep))
    return z_mel
    
      
def mel2hz(z_mel):
    """ Converts a given frequency in the Mel scale to Hz scale.
    
    **Args**: 
        z_mel (Numpy array): Array containing the frequencies in the Mel scale that should be converted.
        
    **Returns**:
        A Numpy array (of same shape as z_mel) containing the converted frequencies.
    """
    f_0 = 0
    f_sp = 200.0/3.0
    brkfrq = 1000.0
    brkpt  = (brkfrq - f_0)/f_sp
    logstep = sp.exp(sp.log(6.4)/27)
    f_hz = sp.where(z_mel < brkpt, f_0 + f_sp*z_mel, brkfrq*sp.exp(sp.log(logstep)*(z_mel-brkpt)))
    return f_hz
    
    
def getValidKeywords(kw, func):
    """ This function returns a dictionary containing the keywords arguments in initial_kw 
        that are valid for function func.
    """
    import inspect
    valid_kw = {}
    invalid_kw = kw.copy()
    args, varargs, varkw, defaults = inspect.getargspec(func)
    for k, v in kw.iteritems():
        if k in args:
            valid_kw[k] = v
            del invalid_kw[k]
    return valid_kw, invalid_kw
    
    
def deltas(x, w=3):
    """ this function estimates the derivative of the 
    """
    if(x.ndim==1):
        y = x.reshape((-1,1)).T
    else:
        y = x   
    if not w%2:
        w -= 1
    hlen = sp.floor(w/2)
    if w==1: # first-order difference:
        win = sp.r_[1,-1]
    else:
        win = sp.r_[hlen:-hlen-1:-1]
    # Extending the input data (avoid border problems)
    extended_x = sp.c_[sp.repeat(y[:,0].reshape((-1,1)),hlen,axis=1), y, 
                       sp.repeat(y[:,-1].reshape((-1,1)),hlen,axis=1)]
    d = scipy.signal.lfilter(win, 1, extended_x, axis=1)
    d = d[:,2*hlen:]
    if x.ndim==1:
        d = d[0,:]    
    return d

def dtft(x, chosen_omega):
    """ Discrete-Time Fourier Transform 
        This function returns the DTFT of x for selected discrete frequencies omega.
    """
    ret = sp.zeros(chosen_omega.shape, dtype='complex')
    for count, omega in enumerate(chosen_omega):
        ret[count] = (x*sp.exp(-1j*omega*sp.arange(0, x.size))).sum()
    return ret
    
def apply_to_window(X, window_size, hop, foo):
    """ This function applies foo to blocks of window_size over the lines of X. The signal
    is mirrored in the borders to avoid problems."""
    ext_size = sp.floor(window_size/2.0)
    aux_X = sp.atleast_2d(X)
    res = sp.zeros(aux_X.shape)
    for line_ind, x in enumerate(aux_X):
        ext_x = sp.concatenate((x[ext_size:0:-1], x, x[-2:-2-ext_size:-1]))
        seg_x = segmentSignal(ext_x, window_size, hop)
        res[line_ind, :] = foo(seg_x, axis=0)
#         for col_ind, frame in enumerate(seg_x.T):
#             res[line_ind, col_ind]  = foo(frame)
    return res
    
def sse(X, filt_len):
    """ This function applies foo to blocks of window_size over the lines of X. The signal
    is mirrored in the borders to avoid problems."""
    if filt_len % 2 == 0: filt_len += 1
    ext_size = sp.floor(filt_len)+3
    ext_x = sp.atleast_2d(X)
    ext_x = sp.concatenate((ext_x, ext_x[-2:-2-ext_size:-1, :]))
    ext_x = scipy.signal.lfilter([1, 1, 1], 1, ext_x, axis=0)/3.
    non_zero = ext_x != 0
    ext_x[non_zero]= 1./ext_x[non_zero]
    fil = sp.ones((filt_len,))
    ext_x = scipy.signal.lfilter(fil, 1, ext_x, axis=0)/float(filt_len)
    result = ext_x[ext_size:, :]
    non_zero = result != 0
    result[non_zero] = 1./result[non_zero]
    return result
    

def sse_decomposition(X, hor_filter_len, ver_filter_len):
    X_hor = sse(X, hor_filter_len) 
    X_col = X - X_hor
    #X_col[X_col<=0] = 10**(-120./20)
    X_col = (sse(X.T, ver_filter_len)).T
    return X_hor, X_col

def median_decomposition(X, hor_filter_len, ver_filter_len):
    """ This function decomposes the matrix X into two: one after applying a median filter
    to the columns and the other the result of applying a median filter to the lines.
    """
    X_hor = apply_to_window(X, hor_filter_len, 1, sp.median) 
    X_col = (apply_to_window(X.T, ver_filter_len, 1, sp.median)).T
    return X_hor, X_col
    

def logistic(x):
    """ Logistic function."""
    return 1. / (1 + sp.exp(-x))
    


def no_sidelobe_window(win_size):
    """ This function returns a windowing function that has no sidelobes. The window has
        size equal to win_size."""
#     a = 1.8
#     b = 0.92
    a = 1.8
    b = 0.01
    rem = False
    if win_size % 2 != 0:
        win_size += 1
        rem = True
    half_size = (win_size)/2.0
    n = sp.absolute(sp.arange(-half_size, half_size))
    aux1 = (1.0 - n / half_size)**a
    aux2 = sp.exp(-b * (n / half_size)**2)
    w = aux1 * aux2
#     h = sp.hanning(win_size)
#     W = sp.fftpack.fft(w, win_size*10)
#     H = sp.fftpack.fft(h, win_size*10)
#     f = sp.fftpack.fftfreq(len(W))
#     plt.plot(f, 20*sp.log10(sp.absolute(W)))
#     plt.plot(f, 20*sp.log10(sp.absolute(H)),'r--')
#     #plt.plot(w)
#     plt.show()
    if rem:
        w = w[0:win_size-1]
    return w