import scipy as sp
import scipy.sparse as ss
from itertools import product, chain
import pickle
import matplotlib.pyplot as plt
import itertools
import time
import copy

class HiddenVariable(object):
    """ A Hidden Random Variable in a Hidden Markov Model.
     
        **Decription**
        ----------------
        This class describes a hidden variable inside a hidden markov model. Each variable
        can assume a given state. Each HiddenVariable should have a name, a state list,
        and a function that returns the probability of going from one state to another.
         
        **Members**:
        ----------------
        - state_list: list containing the possible states of the sequence.
        - name: state name
        - transition probability: probability for going from one state to another
        - dependencies

        **Properties**:
        ----------------
        - This class is iterable: iterating over a variable returns a list of possible
        states.
   
        ----------------
    """
    def __init__(self, name=None, state_list=None):
        self.name = name
        self.state_list = state_list
        
    def __iter__(self):
        i = 0
        while i < len(self.state_list):
            yield self.state_list[i]
            i+=1  
    
    
class HMM_Model(object):
    """ Creates a Hidden Markov Model. 
     
        **Decription**
        ----------------
        This class creates a Hidden Markov Model from a set of hidden variables. Each 
        state must be iterable and possess a finite number of states. Variables are 
        added through the add_state() method. After all states are added, the transitions
        probabilities, and prior probability of each state are computed using by the 
        update() method. If the number of states is greater than sparse_thr (a class
        varibale), then all data is stored in sparse representation, otherwise scipy 
        arrays are employed. After it is created, this model can be employed by any
        inference available in the package (i.e. forward_backward_algorithm). A 
        likelihood function must also be provided to the model. A likelihood function 
        must receive two parameters: an observation and a valid model state.
        
        **Members**:
        ----------------
        - transition: scipy array or scipy sparse matrix (compressed row format)
        - likelihood: function that receives an observation and returns a vector of 
        probabilities associated with each possible state.
        - prior: property that returns the prior probabilities calculated according to 
        a previously set prior function that receives a state and returns its 
        probability.

        **Properties**:
        ----------------
        - prior: read only. Returns the prior probability of each state.

        ----------------
    """
    sparse_thr = 500

    def __init__(self, prior=None, hidden_va_list=None, likelihood=None, transition=None,
                 annotation={}):
        self.transition = None
        self.prior_foo = prior
        if hidden_va_list == None:
            self.hidden_va_list = []
        else:
            self.hidden_va_list = hidden_va_list
        self.likelihood_foo = likelihood
        self.transition_foo = transition
        self.annotation = annotation
    
    def update(self):
        # Creating state lists and setting number of VAs. Creating empty likelihood and 
        # empty transition matrix
        self.__update_lists()
        # Calculating transition probability
        for i, j in product(range(self.n_states), repeat=2):
            si = self.state_list[i]
            sj = self.state_list[j]
            aux_transition = self.transition_foo(self.va_names, si, sj)
            if aux_transition: 
                self.transition[i, j] = aux_transition
        # If sparse representations are being used, the matrices are converted to 
        # compressed row format for efficient multiplication
        if self.n_states > self.sparse_thr:
            self.transition = ss.csr_matrix(self.transition)

    def __update_lists(self):
        # Creating a list of all possible states
        self.state_list = [state for state in product(*self.hidden_va_list)]
        self.va_names = [va.name for va in self.hidden_va_list]
        self.n_states = len(self.state_list)
        if self.n_states > self.sparse_thr:
            self.likelihood_vec_zeros = ss.lil_matrix((self.n_states, self.n_states))
            self.transition = ss.lil_matrix((self.n_states, self.n_states))
        else:
            self.likelihood_vec_zeros = sp.matrix(sp.zeros((self.n_states, self.n_states)))
            self.transition = sp.matrix(sp.zeros((self.n_states, self.n_states)))

    @property
    def prior(self):
        # Calculating prior:
        if self.n_states > self.sparse_thr:
            out = ss.lil_matrix((self.n_states, 1))
        else:
            out = sp.matrix(sp.zeros((self.n_states, 1)))
        # Calculating prior probability
        for i in range(self.n_states):
            aux_prior = self.prior_foo(self.state_list[i])
            if aux_prior:
                out[i, 0] = aux_prior
        if self.n_states > self.sparse_thr:
            out = out.tocsr()
        return out
        
           
    def likelihood(self, observation, n, vec):
        like_vec  = self.likelihood_vec_zeros.copy()
        # Calls the likelihood function for each state given the observation
        if ss.issparse(vec):
            aux = sp.zeros(self.n_states)
            vec_coo = vec.tocoo()
            for i, j, v in itertools.izip(vec_coo.row, vec_coo.col, vec_coo.data):
                like_vec[i, i] = self.likelihood_foo(observation, n, self.state_list[i])        
        for i in range(self.n_states):
            aux_like = self.likelihood_foo(observation, n, self.state_list[i])
            if aux_like:
                like_vec[i, i] = aux_like
        return like_vec


        
    def save(self, filename):
        """ Saves the model in a Numpy compatible file. Should be called after 
        self.update_all has been called. 

        ATTENTION!!!
        The prior and likelihood functions are not saved!
        """
        # Opening output file name
        f = file(filename, 'wb')
        # First saving number of hidden vas:
        pickle.dump(len(self.hidden_va_list), f)
        # First saving each hidden variable data
        for v in self.hidden_va_list:
            pickle.dump(v.state_list, f)
            pickle.dump(v.name, f)
        # Saving model annotations
        pickle.dump(self.annotation, f)
        # Saving transition matrix
        if self.n_states <= self.sparse_thr:
            sp.save(f, self.transition)        
        else:
            sp.save(f, self.transition.data)
            sp.save(f, self.transition.indices)
            sp.save(f, self.transition.indptr)
        # Closing file
        f.close()     

    def load(self, filename):
        """ Saves the model in a Numpy compatible file. Should be called after 
        self.update_all has been called. 

        ATTENTION!!!
        The prior and likelihood functions are not saved!
        """
        # Opening output file name
        f = file(filename, 'r')
        # First saving number of hidden vas:
        n_vas = pickle.load(f)
        # First saving each hidden variable data
        self.hidden_va_list = []
        for i in range(n_vas):
            temp_list = pickle.load(f)
            temp_name = pickle.load(f)
            self.hidden_va_list.append(HiddenVariable(temp_name, temp_list))
        # Loading annotations:
        self.annotation = pickle.load(f)
        # Recreating lists and zeroing transition matrix:
        self.__update_lists()
        # Loading transition matrix:
        if self.n_states <= self.sparse_thr:
            self.transition = sp.load(f)
        else:
            t_data = sp.load(f)
            t_indices = sp.load(f)
            t_indptr = sp.load(f)
            self.transition = ss.csr_matrix((t_data, t_indices, t_indptr), 
                                            shape=(self.n_states, self.n_states))
        # Closing file
        f.close()
         

def forward_algorithm(observations, model):
    """ Forward Algorithm for Inference on a Hidden Markov Model.
     
        **Description**
        ----------------
        This function implements the forward algorithm for inference smoothing of a Markov
        Hidden Mode. The input is an array containing the observations at each time
        instant and the model is an object of the bayes_net.Model class describing the 
        likelihood function (emission probabilities) and the transition matrix. The output
        are the observation probabilities for each state at each time instant. 
         
        **Args**:
        ----------------
        - observations: array-like (currently only tested for 1-D arrays)
        - model: object of the ra.bayes_net.Model class
        
       **Returns**:
        ----------------
       - List containing the probabilities of each state being observed in each time 
       instant given only the observations up to that time instant. 
       The probability vector is normalized to one for each time instant and are
       array-like (possibly sparse).

        ----------------
    """
    alpha = [model.prior]
    alpha[0] = alpha[0] / alpha[0].sum()
    for n, x in enumerate(observations):
        aux = model.transition.T * alpha[n]
        curr_alpha = model.likelihood(x, n, aux) * aux
        curr_alpha /= curr_alpha.sum()
        alpha.append(curr_alpha)
    return alpha    


def backward_algorithm(observations, model):
    """ Backward Algorithm for Inference on a Hidden Markov Model.
     
        **Description**
        ----------------
        This function implements the backward algorithm for inference smoothing of a Markov
        Hidden Mode. The input is an array containing the observations at each time
        instant and the model is an object of the bayes_net.Model class describing the 
        likelihood function (emission probabilities) and the transition matrix. The output
        are the smoothing probabilities of each state for each time instant.
         
        **Args**:
        ----------------
        - observations: array-like (currently only tested for 1-D arrays)
        - model: object of the ra.bayes_net.Model class
        
       **Returns**:
        ----------------
       - List containing the probabilities of each state being observed in each time
       given the data from the last time instant down to the current instant. 
       The probability vector should be normalized to one for each time instant and are
       array-like (possibly sparse).
    """
    beta = [sp.matrix(sp.ones(model.n_states,)).T]
    for n, x in enumerate(observations[::-1]):
        curr_beta = model.transition * model.likelihood(x, n, beta[n]) * beta[n]
        curr_beta /= curr_beta.sum()
        beta.append(curr_beta)
    beta.reverse()
    return beta
       
    
def forward_backward_algorithm(observations, model):
    """ Forward-Backward Algorithm for inference on a HMM model.
     
        **Description**
        ----------------
        This function implements the forward backward algorithm for inference no a Markov
        Hidden Mode. The input is an array containing the observations at each time
        instant and the model is an object of the bayes_net.Model class describing the 
        likelihood function (emission probabilities) and the transition matrix. The output
        are the smoothed probabilities of each state for each time instant.
         
        **Args**:
        ----------------
        - observations: array-like (currently only tested for 1-D arrays)
        - model: object of the ra.bayes_net.Model class
        
       **Returns**:
        ----------------
       - List containing the probabilities of each state being observed in each time 
       instant given all possible observations. 
       The probability vector should be normalized to one for each time instant and is
       array-like (possibly sparse).

    """
    # Compute forward probabilities
    alpha = forward_algorithm(observations, model)
    # Compute backward probabilities
    beta = backward_algorithm(observations, model)
    # Compute smoothed probabilities
    output_prob = []
    for alpha_n, beta_n in zip(alpha, beta):
        if ss.issparse(alpha_n):
            curr_prob = alpha_n.multiply(beta_n)
        else:
            curr_prob = sp.multiply(alpha_n, beta_n)
        curr_prob /= sp.sum(curr_prob)
        output_prob.append(curr_prob)
    del alpha[0]
    del beta[-1]
    return output_prob, alpha, beta
    
    
def viterbi(observations, model):
    """ Viterbi Algorithm for inference on a HMM model.
     
        **Description**
        ----------------
        This function implements the viterbi algorithm for inference on a Markov
        Hidden Mode. The input is an array containing the observations at each time
        instant and the model is an object of the bayes_net.Model class describing the 
        likelihood function (emission probabilities) and the transition matrix. The output
        is the most probable state sequence and its probability.
         
        **Args**:
        ----------------
        - observations: array-like (currently only tested for 1-D arrays)
        - model: object of the ra.bayes_net.Model class
        
       **Returns**:
        ----------------
       - List containing the most probable state sequence.
       - Probability of the chosen state sequence.

        ----------------
    """
    if model.n_states > model.sparse_thr:
        return sparse_viterbi(observations, model)    
    v = []
    v.append(sp.log(model.likelihood(observations[0], 0, model.prior) * model.prior)[:,0])
    path = []
    [path.append([i]) for i in range(model.n_states)]
    log_A_T = sp.log(model.transition.T)
    # Decoding the observation sequence:
    for n, x in enumerate(observations[1:]):
        curr_v = []
        curr_path = []
        for s in range(0, model.n_states):
            p_aux = log_A_T[s, :] + v[-1]
            ind_aux = p_aux.argmax()
            p_aux = p_aux.max()
            curr_v.append(p_aux + sp.log(model.likelihood_foo(x, n, model.state_list[s])))
            curr_path.append(path[ind_aux] + [s])
        path = curr_path
        v.append(curr_v)
    # Backtracking:
    p_final, ind = max([(v[-1][i], i) for i in range(model.n_states)])
    return path[ind], p_final


def sparse_viterbi(observations, model):
    """ Viterbi Algorithm for inference on a Sparse HMM model.
     
        **Description**
        ----------------
        This function implements the viterbi algorithm for inference on a Markov
        Hidden Mode. The input is an array containing the observations at each time
        instant and the model is an object of the bayes_net.Model class describing the 
        likelihood function (emission probabilities) and the transition matrix. The output
        is the most probable state sequence and its probability.
         
        **Args**:
        ----------------
        - observations: array-like (currently only tested for 1-D arrays)
        - model: object of the ra.bayes_net.Model class
        
       **Returns**:
        ----------------
       - List containing the most probable state sequence.
       - Probability of the chosen state sequence.

        ----------------
    """
    v = []
    v = ((model.likelihood(observations[0], 0, model.prior) * model.prior))
    v.data = sp.log(v.data)
    v = v.T
#     print v[-1].nnz
    path = []
    [path.append([i]) for i in range(model.n_states)]
    A = model.transition.copy()
    A.data = sp.ones(A.data.shape, dtype='bool')
    A_t = A.T
    log_A_t = model.transition.T
    log_A_t.data = sp.log(log_A_t.data)
    aux_curr_path = [[] for k in range(model.n_states)]
    # Decoding the observation sequence:
    for n, x in enumerate(observations[1:]):
        curr_v = ss.lil_matrix((1, model.n_states))
        curr_path = list(aux_curr_path)
        alpha = (v * A).tocoo()
        start = time.clock()
        for s in alpha.col:
            p_aux = (A_t[s, :].multiply((log_A_t[s, :] + v))).tocoo()
            p_aux, gar, ind_aux = max([(val, i , j) for val, i, j in itertools.izip(p_aux.data, p_aux.row, p_aux.col)])
            obs = model.likelihood_foo(x, n, model.state_list[s])
            if obs and p_aux:
                temp_p = sp.log(obs) + p_aux
                curr_v[0, ind_aux] = temp_p
                curr_path[s] = path[ind_aux] + [s]
        finish = time.clock()
        path = curr_path
        print n
        v = curr_v.tocsr()
#         raw_input()
    # Backtracking:
    aux_v = v.tocoo()
    p_final, ind, gar = max([(val, i , j) for val, i, j in itertools.izip(aux_v.data, aux_v.row, aux_v.col)])
    return path[ind], p_final

    