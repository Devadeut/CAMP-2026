##################################################################################
# Base Model:
# network_simulator.py -- Network simulator to simulate
# plastic recurrent networks studied in:
# Ref: Sadeh, Clopath and Rotter (PLOS Computational Biology, 2015).
# Emergence of Functional Specificity in Balanced Networks with Synaptic Plasticity.

# CAMP 2026, group 7: Devashri, Ishatpreet, Homna, Supraja, Swali 
#################################################################################


# -----------------------
# --- network simulator function: simulate_network
# simulates a network of N integrate-and-fire neurons with plastic synapses using Exact Integration

# -----------------------
# - takes as the input:
# A: coefficient matrix of the subthreshold dynamics (needed for Exact Integration) []
# v0: vector of initial membrane potential of neurons
# x: sequence of input spike trains for all neurons 
# W0: initial weight matrix
# synapse: the type of recurrent synapses (static / plastic)
# -----------------------
# - returns as the output:
# y: vector of membrane potentials for all neurons across time
# s: vector of spikes (0: no spike, 1: spike) for all neurons across time
# yd_plst, yp_plst: low-pass filtered versions of membrane potential needed for the (V-dependent) plasticity rule 
# y_avg: mean depolarization of the post-synaptic neuron
# W: final weight matrix
# -----------------------


import numpy as np

from params import *


def _rect_(xx):
    return xx * (xx > 0)

def generate_poisson_input(stimulus_tetha, T):
    ''' Poisson generated spike trains
        Input: 
        - stimulus_theta: orientation
        - T: simulation time
        Returns: spike_train    
    '''
    num_stimuli = len(stimulus_tetha)
    bins_per_stimulus = int(T/num_stimuli)
    total_bins = int(T)

    # Init input matrix (N_neurons, T_timebins)
    spike_train = np.zeros((n, total_bins))

    for i in range(n):
        neuron_spike_train = []

        # For each stimulus angle
        for theta in stimulus_tetha:
            # Tuning rate for neuron i
            # tune component of input modulated using orientation (theta)
            if i < ne:
                p_rate = b_rate * (1 + m_exc * np.cos(2 * (theta - po_init[i])))
            else:
                p_rate = b_rate * (1 + m_inh * np.cos(2 * (theta - po_init[i])))
            
            # Generate the Poisson spikes for the duration of this stimulus
            lam = p_rate * dt / 1000.0
            spikes = np.random.poisson(lam, bins_per_stimulus)
            neuron_spike_train.extend(spikes)

        spike_train[i,:] = neuron_spike_train
    
    return spike_train

class LIFNeuron:
    '''Leaky integrate-and-fire neuron population with exact integration.'''

    def __init__(self, A, threshold):
        self.decay = np.exp(np.asarray(A) * dt)
        self.threshold = threshold

    def step(self, pre_Vm, input, I_syn=0):
        V = (self.decay * np.asarray(pre_Vm).reshape(-1)+ input + I_syn ) # Leaky and Integrate step
        spikes = V > self.threshold                                                   # Fire step
        V = V * (V <= self.threshold)
        return V, spikes


class STDP:
    '''V-dependent plasticity rule for recurrent synapses.'''

    def initial_traces(self, V0):
        V0 = np.asarray(V0).reshape(-1)
        yd_trace = Bm_plst * V0 + V0 / tm_plst
        yp_trace = Bp_plst * V0 + V0 / tp_plst
        x_trace = np.zeros_like(V0)
        return yd_trace, yp_trace, x_trace

    def update_traces(self, pre_yd_trace, pre_yp_trace, pre_x_trace, pre_Vm, spikes):
        yd_trace = Bm_plst * pre_yd_trace + pre_Vm / tm_plst
        yp_trace = Bp_plst * pre_yp_trace + pre_Vm / tp_plst
        x_trace = Bx_plst * pre_x_trace + spikes / tx_plst
        return yd_trace, yp_trace, x_trace

    def running_avg_Vm(self, V, t):
        if t == 0:
            return np.zeros(V.shape[0])

        t_avg = 0.1  # seconds
        window = int(t_avg * 1000 / dt)
        t0 = max(t - window, 0)
        return np.mean(V[:, t0:t], axis=1) # average mem potential

    def update_weights(self, W, V, spikes, yd_trace, yp_trace, x_trace, avg_Vm):
        # depression term
        dw_d = -dt * np.outer(spikes, (A_ltd * avg_Vm**2 / 70.0) * _rect_(yd_trace - vth_m))
        # potentiation term
        dw_p = dt * A_ltp * np.outer(x_trace, _rect_(V - vth_p) * _rect_(yp_trace - vth_m))

        # exc to all
        W[0:ne, 0:n] = (W[0:ne, 0:n] + (dw_d[0:ne, 0:n] + dw_p[0:ne, 0:n]) * (W[0:ne, 0:n] != 0))
            # bounds
        W[0:ne] = (W[0:ne] >= w_max) * w_max + (W[0:ne] < w_max ) * W[0:ne] 
        W[0:ne] *= W[0:ne] > 0

        # inh to exc
        W[ne:, 0:ne] = (W[ne:, 0:ne] - dw_p[ne:, 0:ne] - dw_d[ne:, 0:ne])
            # bounds
        W[ne:] = (W[ne:] <= w_max_inh) * w_max_inh + (W[ne:] > w_max_inh) * W[ne:]
        W[ne:] *= W[ne:] <= 0

        return W


def simulate_network(A, v0, x, vth, W0, synapse='static'):
    inputs = np.asarray(x)
    V0 = np.asarray(v0).reshape(-1)
    W = np.copy(W0)

    V = np.zeros(inputs.shape)
    spikes = np.zeros(inputs.shape)
    yd_plst = np.zeros(inputs.shape)
    yp_plst = np.zeros(inputs.shape)
    x_plst = np.zeros(inputs.shape)
    y_avg = np.zeros(inputs.shape)

    neuron = LIFNeuron(A, vth)
    stdp = STDP()

    for t in range(inputs.shape[1]):
        if t == 0:
            pre_Vm = V0
            I_syn = 0
            yd_trace, yp_trace, x_trace = stdp.initial_traces(V0)
        else:
            pre_Vm = V[:, t - 1]
            pre_spikes = spikes[:, t - 1]
            I_syn = pre_spikes @ W
            yd_trace, yp_trace, x_trace = stdp.update_traces(yd_plst[:, t - 1], yp_plst[:, t - 1], x_plst[:, t - 1], pre_Vm, pre_spikes)

        V[:, t], spikes[:, t] = neuron.step(pre_Vm, inputs[:, t], I_syn)
        yd_plst[:, t], yp_plst[:, t], x_plst[:, t] = yd_trace, yp_trace, x_trace

        if synapse == 'plastic':
            y_avg[:, t] = stdp.running_avg_Vm(V,t)
            W = stdp.update_weights(W, V[:, t], spikes[:, t], yd_trace, yp_trace, x_trace, y_avg[:, t])

    return V, spikes, yd_plst, yp_plst, y_avg, W



# ----
