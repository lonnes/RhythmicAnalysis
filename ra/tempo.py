from ra import numeric
from ra import *
import scipy as sp
import exceptions
import copy
from math import exp, sqrt
from collections import namedtuple
from itertools import product
import matplotlib.pyplot as plt


def apply_weigth(periodicity, weight_fun=None):
    if weight_fun != None:
        periodicity.data = periodicity.data * weight_fun(periodicity.bpm)
    return periodicity


def maxNPeaks(periodicity, N=None, return_strength=False):
    """ 
        This function returns the N most likely tempo candidates given the periodicity.
        If N is not specified, all candidates are returned.
    """
    # Selecting peaks in the periodicity vector:
    tempo_ind = numeric.peakPicking(periodicity.data)
    cand_strength = periodicity.data[tempo_ind]    
    cand_tempo = []
    cand_strength_list = []  
    if tempo_ind.ndim == 1:
        ind_cand = sp.argsort(cand_strength, axis=0)
        ind_cand = ind_cand[::-1]
        cand_strength_list = cand_strength[ind_cand][:N].tolist()
        cand_tempo = periodicity.bpm[tempo_ind[:]][ind_cand][:N].tolist()
    else:
        for frame in range(tempo_ind.shape[-1]):
            ind_cand = sp.argsort(cand_strength, axis=0)
            ind_cand = ind_cand[::-1]
            cand_strength_list.append(cand_strength[ind_cand][:N].tolist())
            cand_tempo.append(periodicity.bpm[tempo_ind[:,frame]][ind_cand][:N].tolist())
            if N!=None:
                cand_tempo, cand_strength_list = guarantee_len(cand_tempo, cand_strength_list, N)
    if return_strength:
        return cand_tempo, cand_strength_list
    else:
        return cand_tempo
    
def searchPeakMult(cand_vec, str_vec, mult_vec, inf_tactus_lim=0.2, sup_tactus_lim=1, lim=20e-3):
    str_vec = copy.copy(str_vec)
    cand_vec = copy.copy(cand_vec)
    out_cand = []
    out_str = []
    count = 0
    while len(cand_vec) > count:
        tactus_cand = cand_vec[count]
        if(tactus_cand>=inf_tactus_lim and tactus_cand<=sup_tactus_lim):
            aux_out = [tactus_cand]
            aux_str = [str_vec[count]]
            for mult in mult_vec:
                # Search for tactus in tactus list...
                dist = sp.absolute(mult*tactus_cand-cand_vec)
                aux = dist.argmin()
                #print tactus_cand, tatum_cand_vec[aux]
                if dist[aux] <= lim:
                    aux_str.append(str_vec[aux])
                    aux_out.append(cand_vec[aux])
                    cand_vec = sp.delete(cand_vec, aux)
                    str_vec = sp.delete(str_vec, aux)
            # Deleting candidate itself
            aux = sp.absolute(tactus_cand-cand_vec).argmin()
            cand_vec = sp.delete(cand_vec, aux)
            str_vec = sp.delete(str_vec, aux)            
            out_cand.append(aux_out)
            out_str.append(aux_str)
        else:
            count += 1
    return out_cand, out_str 

        
def guarantee_len(tempo, strength, N):
    """ This function guarantees that the estimated tempo have the chosen length. If the
        length of the input arguments, they are zero-padded to the appropriate length. This
        behavior must happen to guarantee that the result can be stored as arrays in the 
        evalDB file."""
    if N!=1:
        tempo += [0 for i in range(N-len(tempo))]
        strength += [0 for i in range(N-len(strength))]
    return tempo, strength
    
    
class __ClusterManager(object):
    """ Class used to model a cluster"""
    
    class cluster_type(object):
        """ Stores cluster information."""
        def __init__(self, element=None, weight=0):
            self.weight = weight
            self.score = 0
            if element is None:
                self.elements = []
            else:
                self.elements = [element*weight]

        @property
        def interval(self):
            return sp.sum(self.elements)/self.weight

    def __init__(self, cluster_width=None):
        if cluster_width == None:
            self.cluster_width = return_less_than(1.5)
        elif not hasattr(cluster_width, '__call__'):
            self.cluster_width = return_less_than(cluster_width)
        else:
            self.cluster_width = cluster_width
        self.clusters = []
                
    def search_clusters(self, tempo):
        aux_interval = sp.array([i.interval for i in self.clusters])
        # dist = sp.absolute((aux_interval-tempo)) < self.cluster_width
        dist = self.cluster_width(aux_interval,tempo) 
        return sp.nonzero(dist)[0]

    def add_element(self, tempo, weight):
        # Searching for closest cluster:
        dist = self.search_clusters(tempo)
        if dist.size>1:
            dist = dist[0]
        if dist.size!=0:
            self.clusters[dist].elements.append(tempo*weight)
            self.clusters[dist].weight += weight
        else:
            new_cluster = self.cluster_type(tempo, weight)   
            self.clusters.append(new_cluster)
            
    def merge_clusters(self):
        for count, elem in enumerate(self.clusters):
            rem_vec = []            
            dist = self.search_clusters(elem.interval)
            for index in dist:
                if index != count:
                    self.__add_clusters(count, index)
                    rem_vec.append(index)
            self.__del_elements(rem_vec)
            
    def __del_elements(self, rem_vec):
        rem_vec.sort()
        rem_vec.reverse()
        for index in rem_vec:
            self.clusters.pop(index)
                    
    def calc_weights(self, score_fun=None, weight_fun=None):
        if score_fun == None:
            score_fun = lambda x, y, z: x.weight
        if weight_fun == None:
            weight_fun = lambda x: 1.0
        for ind, clu in enumerate(self.clusters):
            self.clusters[ind].score = score_fun(clu, self.clusters, self.cluster_width)
            self.clusters[ind].score *= weight_fun(self.clusters[ind].interval)
            
    def __add_clusters(self, i, j):
        # Adding information from j to i
        self.clusters[i].elements += self.clusters[j].elements
        self.clusters[i].weight += self.clusters[j].weight
 
 
def return_less_than(dist):
    def less_than(x, y):
        return sp.absolute(sp.array(x)-sp.array(y))<dist
    return less_than       
 
        
def prop_width(percentage):
    def less_than(x, y):
        return (sp.absolute(sp.array(x)-sp.array(y))/sp.array(x))<percentage/100.0
    return less_than
    
        
def dixon_score(cl1, clusters, cluster_width):
    n = range(1, 9)
    f = sp.zeros(8)
    f[4:8] = 1
    f[0:4] = range(5, 1, -1)
    score = 0
    print cluster_width
    for count2, cl2 in enumerate(clusters):
        tempo1 = cl1.interval
        tempo2 = cl2.interval
        for mult, w in zip(n, f):
            if abs(tempo1 - tempo2/mult) < cluster_width:
                score += w*clusters[count2].weight
    return score
    
    
def template_score(cl1, clusters, cluster_width):
    """ Peeter template score adapted to work with clusters."""
    beta = [1.0/3, 1.0/2, 1.0, 1.5, 2.0, 3.0]
    alpha = [[-1, 1, 1, -1, 1, -1], [-1, 1, 1, -1, -1, 1], [1, -1, 1, -1, 1, -1]]
    tempo1 = cl1.interval
    max_score = 0
    aux_interval = sp.array([i.interval for i in clusters])
    temp_ind_list = []
    for b in beta:
        tempo2 = tempo1*b
        temp_ind_list.append( __search_cluster_list(tempo2, aux_interval, cluster_width))
    for template in alpha:
        score = 0
        for weight, ind in zip(template, temp_ind_list):
            if ind != None:
                score += weight*clusters[ind].weight
        max_score = max(score, max_score)
    return max_score
    
    
def __search_cluster_list(tempo, tempo_list, cluster_width):
        dist = sp.absolute(tempo-tempo_list)
        ind_min = dist.argmin()
        #if dist[ind_min] < cluster_width:
#         print tempo, tempo_list[ind_min], cluster_width(tempo, tempo_list[ind_min])
        if cluster_width(tempo, tempo_list[ind_min]):
            return ind_min
        else:
            return None

    
def detec_tempo_templates(sim, N=None, min_tempo=40.0, max_tempo=250.0, weight_fun=None,
                          return_strength=False, precision=1.0, compass_criterium = 'max'):
    templates = [[-1, 1, 1, -1, 1, -1], [-1, 1, 1, -1, -1, 1], [1, -1, 1, -1, 1, -1]]
    # templates = [[1, 1, 1, 1, 1]]
    tempo_list = []
    strength_list = []
    sum_list = []
    initial_bpm = maxNPeaks(sim, return_strength=False)
    bpm_fs = sim.bpm[0]-sim.bpm[1]
    actual_bpm = bpm_fs
    if precision != None:
        # Downsampling similarity (speeds-up algorithm)
        target_bpm_fs = precision
        down_factor = int(round(target_bpm_fs/bpm_fs))
        if down_factor > 1:
            b, a = sp.signal.butter(8, 1.0/down_factor)
            sim.data = sp.signal.filtfilt(b, a, sim.data)
            sim.lag = sim.lag[0::down_factor]
            sim.data = sim.data[0::down_factor]
            actual_bpm = down_factor*bpm_fs
    for template in templates:
        tempo_curve, lag = calc_template(sim, template=template, min_lag=60.0/max_tempo, 
                                    max_lag=60.0/min_tempo, threshold=actual_bpm)
#         plt.plot(60./lag, tempo_curve)
#         plt.title('{0}'.format(template))
#         plt.show()
        if weight_fun != None:
            tempo_curve = [str*weight_fun(60.0/l) for str, l in zip(tempo_curve, lag)]
            tempo_curve = sp.array(tempo_curve)
        aux_tempo, aux_strength = maxNPeaks(data.Similarity(data=tempo_curve, lag=lag,
                                              time=sim.time), N, return_strength=True)
        tempo_list.append(aux_tempo)
        strength_list.append(aux_strength)
        sum_list.append(sp.sum(tempo_curve))
    # Finding best list
    max_strength = -sp.inf
    best_template = None
    if compass_criterium == 'max':
        for ind, strength in enumerate(strength_list):
            if strength > max_strength:
                max_strength = strength
                best_template = ind
    elif compass_criterium == 'sum':
        for ind, strength in enumerate(sum_list):
            if strength > max_strength:
                max_strength = strength
                best_template = ind       
    else:
        raise ValueError('Invalid pattern selection value.')
    cand_tempo = tempo_list[best_template]
    cand_strength = strength_list[best_template]
    if N!=None:
        cand_tempo, cand_strength = guarantee_len(cand_tempo, cand_strength, N)
    if return_strength:
        return cand_tempo, cand_strength 
    return cand_tempo


def calc_template(sim, template, min_lag, max_lag, threshold=3):
    beta = [1.0/3, 1.0/2, 1.0, 1.5, 2.0, 3.0]
    # outer loop: from min lag to max lag
    min_ind = sp.flatnonzero(sim.lag>=min_lag)[0]
    max_ind = sp.flatnonzero(sim.lag<=max_lag)[-1]
    output = sp.zeros((max_ind-min_ind+1,))
    min_bpm = sim.bpm[-1]
    max_bpm = sim.bpm[0]
    for ind in range(min_ind, max_ind+1):
        bpm = 60.0/sim.lag[ind]
        aux_bpm = sp.array(beta)*bpm
        for w, weight  in zip(aux_bpm, template):
            if w <= max_bpm and w >= min_bpm:
                nearest_ind = sp.absolute(sim.bpm-w).argmin()
                output[ind-min_ind] += sim.data[nearest_ind]*weight
#             if sp.absolute(w-sim.bpm[nearest_ind]) <= threshold:
#                 output[ind-min_ind] += sim.data[nearest_ind]*weight
    return output, sim.lag[min_ind:max_ind+1]
    

def calc_template_peaks(sim, template, min_lag, max_lag, threshold=1, bpm_list=None):
    beta = [1.0/3, 1.0/2, 1.0, 1.5, 2.0, 3.0]
    # outer loop: from min lag to max lag
    min_ind = sp.flatnonzero(sim.lag>=min_lag)[0]
    max_ind = sp.flatnonzero(sim.lag<=max_lag)[-1]
    output = sp.zeros((max_ind-min_ind+1,))
    #for ind in range(min_ind, max_ind):
    for bpm in bpm_list:
        aux_lag = 60./bpm
        if aux_lag>=min_lag and aux_lag <= max_lag:
            ind = sp.absolute(sim.bpm-bpm).argmin()
            aux_bpm = sp.array(beta)*bpm
            for w, weight  in zip(aux_bpm, template):
                nearest_ind = sp.absolute(sim.bpm-w).argmin()
                if sp.absolute(w-sim.bpm[nearest_ind]) < threshold:
                    output[ind-min_ind] += sim.data[nearest_ind]*weight
    return output, sim.lag[min_ind:max_ind+1]


def resonance_weight(tempo, beta=5.0, t0=138):
    tempo = tempo/60.0
    t0 = t0/60.0
    term1 = ((t0**2 - tempo**2)**2 + beta*(tempo**2))**(-0.5)
    term2 = ((t0**4) + (tempo**4))**(-0.5)
    return (term1 - term2)
    
def gaussian_weight(tempo, d0=0.5, theta=1.4):
    d = 60.0/tempo
    return sp.exp(-0.5*((sp.log2(d/d0)/theta)**2))
    
def rayleigh_weight_function(tempo, d0 = 120):
    """ Rayleigh tempo weighting function."""
    d = 60./tempo
    d /= 11.6e-3
    d0 = 43
    return (d/d0**2)*exp(-((d**2))/(2*(d0**2)))
