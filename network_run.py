##################################################################################
# Base Model:
# network_run.py -- Uses the simulator from network_simulator.py
# and runs a simulation of a plastic recurrent network as in:
# Ref: Sadeh, Clopath and Rotter (PLOS Computational Biology, 2015).
# Emergence of Functional Specificity in Balanced Networks with Synaptic Plasticity.
#
# Author: Devashri + add names as you write code (CAMP 2026, group 7 )
##################################################################################

import numpy as np
import matplotlib.pyplot as pl
import pickle 
from importlib import reload # Python 3 compatibility
import params; reload(params); from params import *
from network_simulator import simulate_network, generate_poisson_input

#################################################################################
# -- generating the weight matrix | J = 0.5 mV EPSP, ne- num of exc, ni - num of inh
# binomial distribution with P_xx prob. and size (x, y)
w0_exc = np.concatenate(( J*np.random.binomial(1, P_ee, (ne,ne)), \
                          J*np.random.binomial(1, P_ei, (ne,ni)) ), 1)  # (ne, ne+ni)
w0_inh = np.concatenate(( -g*J*np.random.binomial(1, P_ie, (ni,ne)), \
                          -g*J*np.random.binomial(1, P_ii, (ni,ni)) ), 1) # (ni, ne+ni)
W0 = np.concatenate((w0_exc , w0_inh)) # (ne+ni, ne+ni) = (n, n)

#################################################################################
# -- before learning (single rientation th)
print('### before plasticity')

spike_train = []
sim_time_test = T
bin = int(sim_time_test/dt) # num of time bins

# Input Generation
spike_train_bp = generate_poisson_input([th], T)
x_ap = np.copy(spike_train_bp)
v0 = np.zeros((1,n)) # (1, 500) init Vm
y_bp, s_bp, ym_plst_bp, yp_plst_bp, y_avg_bp, Wf_bp = \
        simulate_network(A = A, v0 = v0, x = spike_train_bp, vth = vth, W0 = W0, synapse='static')
spike_times_bp = np.where(s_bp[0:n,:] != 0)


#################################################################################
# -- during learning (Multiple random orientation)
print('### within plasticity')

W_blk = W0
spike_times_wp = []
W_blk_tot = []
stim_rng_tot = []

for epoch in range(block_no):
    print(epoch)
    stim_rng = np.random.uniform(0, np.pi, int(stim_no))
    stim_rng_tot.append(stim_rng)
    spike_train_wp = generate_poisson_input(stim_rng, T)
    v0 = np.zeros((1,n)) 
    y, s_wp, ym_plst, yp_plst, y_avg, W_blk = \
       simulate_network(A = A, v0 = v0, x = spike_train_wp, vth = vth, W0 = W_blk, synapse='plastic')
    st = np.where(s_wp[0:n,:] != 0)
    spike_times_wp.append(st)
    W_blk_tot.append(W_blk)

Wf = W_blk

#################################################################################
# -- after learning (Testing phase)
print('### after plasticity')

v0 = np.zeros((1,n)) 
y_ap, s_ap, ym_plst_ap, yp_plst_ap, y_avg_ap, Wf_ap = \
        simulate_network(A = A, v0 = v0, x = x_ap, vth = vth, W0 = Wf, synapse='static')
spike_times_ap = np.where(s_ap[0:n,:] != 0)


#################################################################################
# -- spontaneous activity
spont_act = 0 # set it to 1 to simulate spontaneous activity
if spont_act:
    
    print('### spontaneous activity')

    block_no_sp = 10 

    W_sp = Wf

    W_sp_tot = []
    spk_sp_tot = []
    ## plastic spontaneous
    for epoch in range(block_no_sp):
        print(epoch)
        stim_rng = np.arange(1, stim_no+1)
        #stim_rng_tot.append(stim_rng)
        t_stim = T / len(stim_rng)
        spike_train_wp = []
        #np.random.seed(1234)
        for i in range(n):
            rates = []
            for st in stim_rng:
                p_rate = b_rate/2#*st/stim_no
                rates = rates + np.random.poisson(p_rate*dt/1000., bins//len(stim_rng)).tolist()
            rates = np.array(rates)
            spike_train_wp.append(rates)
        spike_train_wp = np.array(spike_train_wp)

        v0 = np.zeros((1,n)) 

        y, s_sp, ym_plst, yp_plst, y_avg, W_sp = \
           simulate_network(A = A, v0 = v0, x = spike_train_wp, vth = vth, W0 = W_sp, synapse='plastic', inh_ltd=1)
        spk_sp = np.where(s_sp[0:n,:] != 0)
        spk_sp_tot.append(spk_sp)
        W_sp_tot.append(W_sp)

#################################################################################
# -- over-representing cardinal orientations
card_act = 0 # 1 simulates an over-representation of cardinal stimulus orientations
if card_act:
    
    print('### cardinal orientations')

    block_no_cd = 20

    W_cd = W0
    W_cd_tot = []
    spk_cd_tot = []
    stim_rng_tot_cd = []
    ## plastic spontaneous
    for epoch in range(block_no_cd):
        print(epoch)
        stim_rng = np.concatenate( (np.random.uniform(0, np.pi, int(stim_no/2)), np.ones(int(stim_no/4))*0., np.ones(int(stim_no/4))*np.pi/2) )
        np.random.shuffle(stim_rng)
        stim_rng_tot_cd.append(stim_rng)
        t_stim = T / len(stim_rng)
        x_cd = []
        for i in range(n):
            rates = []
            for st in stim_rng:
                if i < ne: p_rate = b_rate*(1+m_exc*np.cos(2*(st - po_init[i])))
                else: p_rate = b_rate*(1+m_inh*np.cos(2*(st - po_init[i])))
                rates = rates + np.random.poisson(p_rate*dt/1000., bins//len(stim_rng)).tolist()
            rates = np.array(rates)
            x_cd.append(rates)
        x_cd = np.array(x_cd)

        v0 = np.zeros((1,n)) 

        y, s_cd, ym_plst, yp_plst, y_avg, W_cd = \
           simulate_network(A = A, v0 = v0, x = x_cd, vth = vth, W0 = W_cd, synapse='plastic')
        spk_cd = np.where(s_cd[0:n,:] != 0)
        spk_cd_tot.append(spk_cd)
        W_cd_tot.append(W_cd)

#################################################################################
# -- saving the results

res_save = 1
if res_save:
    results = {}
    results['W0'] = W0

    results['spike_times_bp'] = spike_times_bp
    results['spike_times_wp'] = spike_times_wp
    results['spike_times_ap'] = spike_times_ap
    results['W_blk_tot'] = W_blk_tot
    results['stim_rng_tot'] = stim_rng_tot
    
    if spont_act:
        results['spk_sp_tot'] = spk_sp_tot
        results['W_sp_tot'] = W_sp_tot
    else:
        results['spk_sp_tot'] = []
        results['W_sp_tot'] = []
        
    if card_act:
        results['spk_cd_tot'] = spk_cd_tot
        results['W_cd_tot'] = W_cd_tot
        results['stim_rng_tot_cd'] = stim_rng_tot_cd
    else:
        results['spk_cd_tot'] = []
        results['W_cd_tot'] = []
        results['stim_rng_tot_cd'] = []

    with open('results', 'wb') as fl:       
        pickle.dump(results, fl)

# ------
