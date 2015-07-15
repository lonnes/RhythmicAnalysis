import scipy as sp
from scipy.linalg import norm as norm_p
import scipy.stats as st
import bayes_net as bn
import matplotlib.pyplot as plt
from ra.numeric import logistic


def create_tatum_model(max_tatum, min_period, max_period, sigma_c):
    if not(sigma_c % 2):
        sigma_c += 1
    window_taps = sp.hanning(sigma_c + 2) # 2 boundaries value of hanning are always zero.
    window_taps /= sp.sum(window_taps)
    h_win_len = (sigma_c + 1) / 2

    def ind_foo(x):
        if abs(x) < h_win_len:
            return window_taps[x + h_win_len]
        else:
            return 0.0
            
    def tatum_trans(names, si, sj):
        # si: current state, sj: future state
        if si[0] and sj[2] == (si[1] + 1):
            p_t = 1.0
        elif si[0] and sj[2] == min_period and (si[1] + 1) < min_period:
            p_t = 1.0
        elif si[0] and sj[2] == max_period and (si[1] + 1) > max_period:
            p_t = 1.0
        elif si[2] == sj[2]:
            p_t = 1.0
        else:
            p_t = 0.0
        if p_t != 0.0:
            if si[0] and sj[1] == 0:
                p_c = 1.0
            elif (not si[0]) and (sj[1] == si[1] + 1):
                p_c = 1.0
            elif (not si[0]) and (sj[1] == 0) and (si[1] == max_tatum):
                p_c = 1.0
            else:
                p_c = 0.0
        if p_t != 0.0 and p_c != 0.0:
            aux_p_i = ind_foo(sj[1] - sj[2])
            if sj[0]:
                p_i = aux_p_i
            else:
                p_i = 1.0 - aux_p_i
            return p_c * p_t * p_i
        else:
            return 0.0

    tatum_indicator = bn.HiddenVariable(state_list=[False, True], name='ind_t')
    tatum_counter = bn.HiddenVariable(state_list=range(max_tatum + 1), name='counter_t')
    tatum_period = bn.HiddenVariable(state_list=range(min_period, max_period + 1), 
                                     name='period_t')
    model = bn.HMM_Model(hidden_va_list = [tatum_indicator, tatum_counter, tatum_period], 
                         transition=tatum_trans)
    return model
        
    

def tatum_observation_model(feature, cand_beat, p=8, log_scale=40, log_thr=0.2):
    # Normalizing feature:
    norm_feat = normalize_features(feature.data, cand_beat * 4, p=p)
    # Applying logistic function:
    p_obs = logistic(log_scale * (norm_feat - log_thr))
    plt.plot(p_obs)
    plt.show()
    def tatum_likelihood(observation, n, state):
        if state[0]:
            return p_obs[n]
        else:
            return 1.0 - p_obs[n]
    return tatum_likelihood
            

def get_prior_tatum(beat_period, max_tatum, min_period, max_period):
    tatum_conv = sp.array([1./2, 1./3, 1./4, 1., 1./6, 3./2, 2./3, 1./9])
    #tatum_conv = sp.array([1./2, 1./3])
    uniform = 1./len(tatum_conv)
    tatum_cand = sp.array(tatum_conv * beat_period, 'int')
    def tatum_prior(state):
        if (not state[0]) and sp.array(state[2] == tatum_cand).sum() and state[1] <= state[2]:
            return uniform / (max_tatum + 1)
        else:
            return 0.0
    return tatum_prior


def get_prior_beat(beat_period, max_tatum, min_period, max_period):
    tatum_conv = sp.array([1./2, 1./3, 1./4, 1., 1./6, 3./2, 2./3, 1./9])
    #tatum_conv = sp.array([1./2, 1./3])
    uniform = 1./len(tatum_conv)
    tatum_cand = sp.array(tatum_conv * beat_period, 'int')
    def tatum_prior(state):
        if (not state[0]) and sp.array(state[2] == tatum_cand).sum() and state[1] <= state[2]:
            return uniform / (max_tatum + 1)
        else:
            return 0.0
    return tatum_prior


def create_beat_model(max_tatum, min_period, max_period, sigma_c):
    if not(sigma_c % 2):
        sigma_c += 1
    window_taps = sp.hanning(sigma_c + 2) # 2 boundaries value of hanning are always zero.
    window_taps /= sp.sum(window_taps)
    h_win_len = (sigma_c + 1) / 2
    max_beat = 3

    def ind_foo(x):
        if abs(x) < h_win_len:
            return window_taps[x + h_win_len]
        else:
            return 0.0
            
    def beat_trans(names, si, sj):
        # si: current state, sj: future state
        # Starting with tatum states:
        if si[0] and sj[2] == (si[1] + 1):
            p_t = 1.0
        elif si[0] and sj[2] == min_period and (si[1] + 1) < min_period:
            p_t = 1.0
        elif si[0] and sj[2] == max_period and (si[1] + 1) > max_period:
            p_t = 1.0
        elif si[2] == sj[2]:
            p_t = 1.0
        else:
            p_t = 0.0
        if p_t != 0.0:
            if si[0] and sj[1] == 0:
                p_c = 1.0
            elif (not si[0]) and (sj[1] == si[1] + 1):
                p_c = 1.0
            elif (not si[0]) and (sj[1] == 0) and (si[1] == max_tatum):
                p_c = 1.0
            else:
                p_c = 0.0
        # Adding beat model:
        if p_t != 0.0 and p_c != 0.0:
            aux_p_i = ind_foo(sj[1] - sj[2])
            if sj[0]:
                p_i = aux_p_i
                if (not si[3]) and (sj[5] == si[5]): # Period rules
                    p_b_t = 1.0
                elif si[3] and (sj[5] == si[4] + 1):
                    p_b_t = 1.0
                else:
                    return 0.0
                if si[3] and (sj[4] == 0):
                    p_b_c = 1.0
                elif (not si[3]) and (sj[4] == si[4] + 1) and (si[4] != max_beat):
                    p_b_c = 1.0
                elif (not si[3]) and (sj[4] == 0) and (si[4] == max_beat):
                    p_b_c = 1.0
                else:
                    p_b_c = 0.0
                p_b = p_b_c * p_b_t
            else:
                p_i = 1.0 - aux_p_i
                if sj[3] and (si[4] == sj[4]) and (si[5] == sj[5]):
                    # Only trivial updates are allowed to the beat when no tatum was
                    # detected
                    p_b = 1.0
                else:
                    return 0.0
            return p_c * p_t * p_i * p_b
        else:
            return 0.0

    tatum_indicator = bn.HiddenVariable(state_list=[False, True], name='ind_t')
    tatum_counter = bn.HiddenVariable(state_list=range(max_tatum + 1), name='counter_t')
    tatum_period = bn.HiddenVariable(state_list=range(min_period, max_period + 1), 
                                     name='period_t')
    beat_indicator = bn.HiddenVariable(state_list=[False, True], name='ind_b')
    beat_counter = bn.HiddenVariable(state_list=[0, 1, 2], name='counter_b')
    beat_period = bn.HiddenVariable(state_list=[2, 3], name='period_t')
    model = bn.HMM_Model(hidden_va_list=[tatum_indicator, tatum_counter, tatum_period, 
                        beat_indicator, beat_counter, beat_period], transition=beat_trans)
    return model


def def_norm_feat_gen(data, max_period, p):
    if not(max_period  % 2):
        max_period += 1
    ext_len = (max_period - 1) / 2
    ext_data = data[1:ext_len + 1][::-1]
    ext_data = sp.append(ext_data, data)
    ext_data = sp.append(ext_data, data[-2:-ext_len - 2:-1])

    def aux(i, win_size):
        fac = (win_size % 2)
        h_len = (win_size / 2)
        aux = norm_p(ext_data[i - h_len + ext_len : i + ext_len + h_len + fac], ord=p)
        return ext_data[i + ext_len] / max(aux, 1e-20)

    return aux


def normalize_features(data, win_len, p):
    foo = def_norm_feat_gen(data, win_len, p)
    out = data.copy()
    for i in range(data.size):
        out[i] = foo(i, win_len)
    return out
    

def create_simple_pattern_model(pattern_len, period, sigma_c):
    """ Creates a simple HMM pattern model. Parameters are:
        pattern_len: number of tatums inside the pattern.
        period: estimated period in frames.
        sigma_c: half the size of window for detecting tatums."""
    # Creating window function:    
    window_taps = sp.hanning(2 * sigma_c + 3) # 2 boundaries value of hanning are always zero.
    window_taps /= sp.sum(window_taps)
    h_win_len = (len(window_taps) - 1) / 2
    def ind_foo(x):
        if abs(x) >= period - sigma_c - 1:
            return window_taps[x - period + 1 + h_win_len]
        else:
            return 0.0

    def pattern_trans(names, si, sj):
        # si: current state, sj: future state, [0] = counter, [1] = pattern_index
        if  si[0] == period + sigma_c - 1 and sj[0] == 0:
            p_c = 1.0
        elif si[0] != period + sigma_c - 1 and sj[0] == 0:
            p_c = ind_foo(si[0])
        elif sj[0] == si[0] + 1:
            p_c = 1.0 - ind_foo(si[0])
        else:
            p_c = 0.0
        if si[0] == 0 and sj[1] == si[1] + 1:
            p_t = 1.0
        elif si[0] != 0 and sj[1] == si[1]:
            p_t = 1.0
        elif si[0] == 0 and si[1] == pattern_len - 1 and sj[1] == 0:
            p_t = 1.0
        else:
            p_t = 0.0
        return p_c * p_t
    
    pattern_index = bn.HiddenVariable(state_list=range(pattern_len), name='pattern_index')
    tatum_counter = bn.HiddenVariable(state_list=range(period + sigma_c), name='tatum_c')
    model = bn.HMM_Model(hidden_va_list=[tatum_counter, pattern_index], 
                         transition=pattern_trans)
    return model


def gen_pattern(tatum_pattern, period):
    """ Generates a rhythmic pattern given the accentuation pattern along the tatum and its 
        period in samples. """
    p_len = len(tatum_pattern) * period
    pattern = sp.zeros((p_len,))
    pattern[0::period] = tatum_pattern
    pattern[1::period] = tatum_pattern
    pattern[2::period] = tatum_pattern
    pattern[-1::-period] = tatum_pattern
    pattern[-2::-period] = tatum_pattern
    return pattern


def get_uniform_prior(model):
    num_states = model.n_states
    def uniform_prior(state):
        return 1. / num_states
    return uniform_prior


def get_pattern_prior(model):
    num_tatums = len(model.hidden_va_list[0].state_list)
    def uniform_prior(state):
        if state[1] == 0:
            return 1. / num_tatums
        else:
            return 0.
    return uniform_prior


def gen_obs_model_simple_pattern(pattern, sigma_o):
    """ This function creates an observation model for rhythmic patterns. An accentuation 
         pattern defined in terms of tatum. A tatum period in samples should also be 
         provided."""

    gauss = st.norm(scale=sigma_o) # zero-mean gaussian, with std = sigma_c

    def pat_likelihood(observation, n, state):
        """ Simple observation. Assumes that the accentuation should be close to the 
        pattern."""
        if state[0] == 0: # Tatum should be observed.
            curr_pat = pattern[state[1]] # Expected occurance 
            p = gauss.pdf(curr_pat - observation)
        else: # Tatum should not be observed.
            p = gauss.pdf(0.0 - observation)
        return p
    return pat_likelihood
        
        
        
    


