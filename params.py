##################################################################################
# Base Model:
# params.py -- Default set of parameters in (Table 1 of):
# Ref: Sadeh, Clopath and Rotter (PLOS Computational Biology, 2015).
# Emergence of Functional Specificity in Balanced Networks with Synaptic Plasticity.
#
# CAMP 2026, group 7: Devashri, Ishatpreet, Homna, Supraja, Swali 
##################################################################################

import numpy as np


##############################
###### ---- Default parameters

# -- stimulation params
block_no = 40 # number of batches for learning
stim_no = 20. # number of stimuli with different orientations in each batch

trial_time = .1 * 1000 # presentation time of each orientation (ms)
T = stim_no * trial_time # total simulation time for each batch (ms)
dt = 1  # time resolution (ms)
bins = int(T/dt)     # NUM OF TIME BINS?
t = np.arange(0, T, dt)

# -- network params
n = 500 # total number of neurons
f = .8 # fraction of excitatory neurons
ne = int(f*n) # number of exc neurons
ni = n - ne   # number of inh neurons

# connection probability
P_ee = .3 # exc to exc
P_ei = .3 # exc to inh
P_ie = 1. # inh to exc
P_ii = 1. # inh to inh

J = .5 # EPSP (mV)
g = 8. # inhibition dominance ratio (IPSP = -g EPSP)

# -- single neuron (Leaky Integrate-and-Fire) params
tm = 20. # membrane time constant (ms)
vth = 20. # threshold voltage (mV)

# coefficient matrix of the subthreshold dynamics (needed for Exact Integration)
A = -1./np.concatenate((tm*np.ones(ne) , tm*np.ones(ni))) 
#B = np.exp(A*dt)

# -- feedforward input
b_rate = 2000. # aggregare baseline rate of the ffw input to a neuron
m = .2 # modulation ratio of the input
m_exc = m # exc ffw modulation ratio
m_inh = m/10 # inh ffw modulation ratio

# initial preferred orientations
po_init = np.concatenate(( np.arange(0, np.pi, np.pi/ne) , np.arange(0, np.pi, np.pi/ni) ))
# stimulus orientation (before and after plasticity)
th = np.pi/2

# -- plasticity params
tm_plst = 10 # tau - (ms)
tp_plst = 7  # tau + (ms)
tx_plst = 15 # tau x (ms)

Bm_plst =  np.exp((-1/tm_plst) *dt)  
Bp_plst =  np.exp((-1/tp_plst) *dt) 
Bx_plst =  np.exp((-1/tx_plst) *dt) 

w_min = 0 # lower bound on all weights
w_max = 2 # upper bound on exc weights
w_max_inh = -5 # lower bound on inh weights

A_ltd = 14.e-5 # (mV-1)
A_ltp = 8.e-5  # (mV-2)

vth_m = -20. # (mV)
vth_p = 7.5 # (mV)

# add structural plasticity params
# Structural plasticity
sp_structural = True    # master on/off
prune_thresh = 0.02     # prune if weight < this (for E->E) # change this later to a parameter to fine-tune
grow_seed = 0.01        # seed weight for grown synapse
p_sp = 0.01             # base growth probability
theta_grow = 0.5        # calcium/activity threshold for growth (tune)
grow_sig_slope = 10.0   # steepness of sigmoid used for growth probability
only_EE_structural = True
allow_inh_structural = False

# -----
