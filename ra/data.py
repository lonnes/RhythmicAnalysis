""" 
    This is package contains several algorithms for Tempo Analysis and Beat tracking in Python.

    Currently this package contains the functions necessary to estimate the tempo.

 """
import os
import scipy as sp
import scikits.audiolab as al
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
from matplotlib.ticker import FuncFormatter, FixedLocator, MaxNLocator, MultipleLocator
import exceptions

            
class Signal(object):
    """
        This class models a time-varying signal (either real or complex). 
                
        The information is stored as a Numpy array.
        
        The following operations are currently supported: loading the data from a file, plotting the data, and playing the data. Depending of the array dimension different operations are available.
        
    """
#     data = DataClass(test_nd_ndarray, sp.zeros((0,)))
#     fs = DataClass(test_positive, 1)
    
    default_xlabel = "Time (s)"
    default_ylabel = "Signal"

    # Number of time samples. Equivalent to the last dimension of self.data.	
    @property
    def num_samples(self):
        return self.data.shape[-1]

    # Period between adjacent frames in seconds:
    @property
    def period(self):
        return 1.0/self.fs

    # Duration in signals
    @property
    def duration(self):
        return self.num_samples*self.period
        
    def __init__(self, data = sp.zeros((0,)), fs=1):
        """The signal can be initialized by a numpy uni-dimensional array. """

        self.data = data
        self.fs = fs
        self.name = "time signal"
        
    def plot(self):
        "Plots the signal using MatPlotLib."
        t = sp.linspace(0,(sp.size(self.data)-1.0)/self.fs,sp.size(self.data))       
        ax = plt.plot(t,self.data)
        plt.xlabel(self.default_xlabel)
        plt.ylabel(self.default_ylabel)
        return plt.gca()
        
    def __add__(self, other):
        """Add overload"""
        if(other.fs == self.fs):
            aux = self
            aux.data = self.data+other.data
            return aux
        raise TypeError
    
class AudioSignal(Signal):

    def __init__(self,data=sp.zeros((0,)),fs=0,filename="",small_footprint=False, 
                 mix_opt='l'):
        """ """
        super(AudioSignal,self).__init__(data, fs)
        if(filename != ""):
            self.load_audiofile(filename, small_footprint, mix_opt)

    def load_audiofile(self, filename, small_footprint=False, mix_opt='l'):
        """Method description"""
        snd_file = al.Sndfile(filename,'r')
        aux_file = al.Sndfile(filename,'r')
        name_ext = os.path.basename(filename)
        name = os.path.splitext(name_ext)[0]
        self.encoding = snd_file.format.encoding
        self.file_format = snd_file.format.file_format
        self.filename = name
        self.fs = snd_file.samplerate
        if small_footprint and self.encoding=='pcm16':
            # Reading as flot and converting to int16: avoids bug for stereo files.
            temp_data = snd_file.read_frames(snd_file.nframes)
            temp_data = sp.array(temp_data*32768.0, dtype=sp.int16)
        else:
            temp_data = snd_file.read_frames(snd_file.nframes)
        if temp_data.ndim != 1: # For multi-channel audio, only one channel is loaded
            if mix_opt == 'l':
                temp_data = temp_data[:,0]  
            elif mix_opt == 'r':
                temp_data = temp_data[:,1]  
            elif mix_opt == 'dm':
                temp_data = (temp_data[:,0] +  temp_data[:,1]) / 2.0
            else:
                raise ValueError("Invalid channel selection.")
        self.data = temp_data
        snd_file.close()
        
    def play(self):
        al.play(self.data,self.fs)
       

class Feature(Signal):
    """ This class serves as a base class for the storage of features of a Signal.
    
        The features are stored as a 2-D Numpy array inside the data member of this class.
        The data member has a decorator in order to ensure that its type.
    """
    # Number of features. Equivalent to the number of lines in self.data.	
    @property
    def num_features(self):
        if self.data.ndim == 1:
            return 1
        return self.data.shape[0]
    def __init__(self,data=sp.zeros((0,0)), fs=1, signal=None, time_index=None, 
                 feature_index=None, name="unkown", feat_axis_label = "Feature number"):
        """Init method. Sets the data and sampling rate of the Feature."""
        self.data = data
        self.fs = fs
        self.signal = signal
        self.feat_axis_label = feat_axis_label
        self.time_axis_unit = "Time (s)"
        self.name = name
        # Creating time index:
        if(time_index is None):
            self.time_index = sp.arange(self.num_samples)/float(self.fs)
        else:
            if(time_index.shape[0]!=self.num_samples):
                raise ValueError(
                            "Time indices must have the same length as self.num_samples.")
            self.time_index = time_index
        # Creating frequency index:
        if(feature_index is None):
            self.feature_index = sp.arange(self.num_features)
        else:
#             if(feature_index.shape[0]!=self.num_features):
#                 raise ValueError(
#                            "Feature indices must have the same length as self.num_feats.")
            self.feature_index = feature_index
            
    def getFrame(self, frame):
        """ Returns all feature-values for a given frame. Frame can be an integer type or
            floating point precision. If integer it is assumed to be the frame number, if 
            floating point it is assumed to be a time in seconds.
        """      
        if isinstance(frame, int):
            frame = frame
        elif isinstance(frame, float):
            frame  = round(frame/self.fs)
        else:
            raise TypeError("Frame must be an integer or a float.")
        return self.data[:,frame]
        
    def getSignal(self, feature_number):
        """ Returns a time varying onde-dimensional signal containing a reference to the
            chosen feature number. 
        """
        if self.data.ndim==1:
            return self
        return Signal(self.data[feature_number, :], self.fs)
        
    def plot(self, transform = None, positive_only = True):
        """ Plots all features."""      
        if transform == None: 
            transform = lambda x: x
        if(positive_only):
            ind = self.feature_index>=0
        else:
            ind = sp.arange(self.feature_index.shape[0])
        fig = plt.imshow(transform(self.data[ind, :]), origin = 'lower', 
                         aspect = 'auto', interpolation='nearest')
        ax = plt.gca()
        
        def feat_formatter(x, pos):
            'The two args are the value and tick position'
            x_int = (round(x))
            x_int = sp.maximum(x_int, 0.0)
            x_int = sp.minimum((self.feature_index[ind]).size-1, x_int)
            x_int = abs(x_int)
            return '%3.2f' % (self.feature_index[ind][x_int]/1000.0)
            
        def time_formatter(x, pos):
            'The two args are the value and tick position'
            x_int = (round(x))
            x_int = sp.maximum(x_int, 0.0)
            x_int = sp.minimum((self.time_index).size-1, x_int)
            x_int = abs(x_int)
            return '%3.0g' % (round(self.time_index[x_int]*100)/100.0)
            
        feat_format = FuncFormatter(feat_formatter)
        time_format = FuncFormatter(time_formatter)
        ax.yaxis.set_major_formatter(feat_format)
        ax.xaxis.set_major_formatter(time_format)
#         xLoc = MultipleLocator(base=10.0)
#         yLoc = MultipleLocator(base=10.0)
#         xLoc.set_scientific(True)
#         yLoc.set_scientific(True)
#         ax.xaxis.set_major_locator(xLoc)
#         ax.yaxis.set_major_locator(yLoc)
#         xLoc.refresh()
#         yLoc.refresh()
        plt.xlabel(self.time_axis_unit)
        plt.ylabel(self.feat_axis_label)
        
class MusicData(object):
    """ This object contains all data and meta-data related to a music signal."""

    
class Similarity(Feature):
    """
        This class models the similarity function of a audio signal. The similarity function
        has a value exhibits the self-similarity of an audio signal for different time
        instants.
    """
    default_xlabel = "BPM"
    default_ylabel = "Similarity"
    def __init__(self, data = None, lag = None, time=0, feat_id=None):
        """ TODO Documentation"""
        self.default_xlabel = "BPM (s)"
        self.default_ylabel = "Signal"
        if data is None:        
            data = sp.zeros((0,))
        if lag is None:
            lag = sp.zeros((0,))
        if feat_id is None:
            feat_id = 'unknown'
        self.data = data    
        self.lag = lag
        self.time = time
        self.feat_id = [feat_id]
        
    def plot(self, xlabel=default_xlabel, ylabel=default_ylabel):
        """ Plots the periodicity."""
        if self.data.ndim == 1:
            fig = plt.plot(self.bpm, self.data)
        elif self.data.ndim == 2:
            for x in self.data:
                fig = plt.plot(self.bpm, x)
        else:
            raise AttributeError("Data must be either a 1-D or 2-D numpy array.")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return fig
        
    def __add__(self, other):
        """ Adds two similarity vectors."""
        if self.data.shape != other.data.shape:
            raise ValueError("The similarities must have the same dimension.")
        if sp.all(self.lag == other.lag) and sp.all(self.time == other.time):
            return Similarity(self.data+other.data, self.lag, self.time)
        else:
            raise ValueError("Both Similairties must have the same lag and time stamps.")
            
    def __iadd__(self, other):
        """ Adds two similarity vectors."""
        if self.data.shape != other.data.shape:
            raise ValueError("The similarities must have the same dimension.")
        if sp.all(self.lag == other.lag) and sp.all(self.time == other.time):
            self.data += other.data
            return self
        else:
            raise ValueError("Both Similairties must have the same lag and time stamps.")
    
    def aggregate_feat(self, other):
        if self.data.shape[:2] != other.data.shape[:2]:
            raise ValueError("The similarities must have the same dimension.")
        if sp.all(self.lag == other.lag) and sp.all(self.time == other.time):
            self.data, temp_data = sp.atleast_3d(self.data, other.data)
            self.data = sp.concatenate((self.data, temp_data), axis=2)
            self.feat_id += other.feat_id
        else:
            raise ValueError("Both Similairties must have the same lag and time stamps.")        

    @property
    def bpm(self):
        """Get the bpm for each similarity element."""
        return 60.0/self.lag
        
    def getFeatSim(self, ind):
        if self.data.ndim == 3:
            return Similarity(self.data[:,:,ind], self.lag, self.time)
        else:
            return self
            
    @property
    def num_features(self):
        if self.data.ndim == 3:
            return self.data.shape[-1]
        return 1
        
    @property
    def period(self):
        return self.lag[1] - self.lag[0]

if __name__ == "__main__":
    pass