""" This module contains scripts for evaluation of algorithms."""

from scipy import mean, std
import exceptions
import datetime
import ra.numeric
import scipy as sp
                      
def pscore(reference_tempo, estimated_tempo_list, tol=8):
    """ This function calculates the P-Score of a pair of tempo estimates.
         Definition of P-Score according to MIREX."""
    try:
        # Sorting estimated tempo list. The reference tempo should already be sorted 
        # from lowest bpm to highest bpm.
        estimated_tempo_list.sort()
        # Checking if first tempo is within the tolerance (i.e. the algorithm
        # gessed the first tempo correctly)
        tt1 = tol/100.0 > abs((estimated_tempo_list[0]-reference_tempo[0])/float(reference_tempo[0]))
        tt2 = tol/100.0 > abs((estimated_tempo_list[1]-reference_tempo[1])/float(reference_tempo[1]))
        # Final score
        p = tt1*reference_tempo[2] + (1 - reference_tempo[2])*tt2
        return p
    except IndexError:
        raise AttributeError("""Reference tempo should have at least 3 elements and 
                                estimated tempo should have at least 2.""")


def accuracy1(reference_tempo, estimated_tempo, tol=4):
    """ This function calculates the Accuracy1 figure of merit for tempo estimates. This 
    metric considers the tempo correct iff it within a 4 % tolerance window around the
    annotated tempo."""
    if type(reference_tempo) == type([]):
        # Considering the tempo in the ISMIR format
        if len(reference_tempo)>1:
            if reference_tempo[2]>0.5:
                reference_tempo = reference_tempo[0]
            else:
                reference_tempo = reference_tempo[1]
        else:
            reference_tempo = reference_tempo[0]
    if type(estimated_tempo) == type([]):
        # Considering the tempo in the ISMIR format
        estimated_tempo = estimated_tempo[0]
    # Checking if the error is inside the tolerance window:
    output = tol/100.0 > abs((estimated_tempo-reference_tempo)/float(reference_tempo))
    return output     
    
    
def accuracy2(reference_tempo, estimated_tempo, tol=4):
    """ This function calculates the Accuracy2 figure of merit for tempo estimates. This 
    metric considers the tempo correct iff it within a 4 % tolerance window around the
    annotated tempo."""
    mult = [1, 2, 3, 0.5, 1.0/3]
    if type(reference_tempo) == type([]):
        # Considering the tempo in the ISMIR format
        if len(reference_tempo)>1:
            if reference_tempo[2]>0.5:
                reference_tempo = reference_tempo[0]
            else:
                reference_tempo = reference_tempo[1]
        else:
            reference_tempo = reference_tempo[0]
    if type(estimated_tempo) == type([]):
        # Considering the tempo in the ISMIR format
        estimated_tempo = estimated_tempo[0]
    # Checking if the error is inside the tolerance window:
    result = []
    result = [abs((m*estimated_tempo)-reference_tempo) for m in mult]
    result = min(result)
    output = tol/100.0 > result/float(reference_tempo)
    return output                               


def detecPeakExistance(tempo, sim, tol=4):
    """Checks if the desired tempo exists in the similarity function."""
    mult = [1., 2., 0.5, 3., 1./3]
    out = []
    # Formatting reference data
    if type(tempo) == type([]):
        # Considering the tempo in the ISMIR format
        if len(tempo)>1:
            if tempo[2]>0.5:
                tempo = tempo[0]
            else:
                tempo = tempo[1]
        else:
            tempo = tempo[0]
    # Detecting peaks
    peaks = ra.numeric.peakPicking(sim.data)
    # BPM of detected peaks
    str = sim.data[peaks]
    order = str.argsort()[::-1]
    peak_bpm = sim.bpm[peaks][order]
    # Checking if there is a peak within the tolerance
    for m in mult:
        ind = (sp.absolute(peak_bpm - (m*tempo))/(m*tempo) < tol/100.0)
        if True in ind:
            id = ind.nonzero()[0][0]
            out.append(id+1)
        else:
            out.append(0)
    return out