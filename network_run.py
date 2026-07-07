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
# binary connectivity mask: 1 = connection exists, 0 = absent
C0 = (W0 != 0).astype(float)
C_blk = C0.copy()
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
# run with current connectivity mask C0
y_bp, s_bp, ym_plst_bp, yp_plst_bp, y_avg_bp, Wf_bp, C_bp = \
    simulate_network(A = A, v0 = v0, x = spike_train_bp, vth = vth, W0 = W0, C0=C0, synapse='static')
spike_times_bp = np.where(s_bp[0:n,:] != 0)


#################################################################################
# -- during learning (Multiple random orientation)
print('### within plasticity')

W_blk = W0
spike_times_wp = []
W_blk_tot = []
C_blk_tot = []
pruned_counts = []
grown_counts = []
total_conn_counts = []
stim_rng_tot = []

for epoch in range(block_no):
    print(epoch)
    stim_rng = np.random.uniform(0, np.pi, int(stim_no))
    stim_rng_tot.append(stim_rng)
    spike_train_wp = generate_poisson_input(stim_rng, T)
    v0 = np.zeros((1,n)) 
    # pass current connectivity mask C_blk into the simulator and receive updated mask back
    y, s_wp, ym_plst, yp_plst, y_avg, W_blk, C_blk = \
       simulate_network(A = A, v0 = v0, x = spike_train_wp, vth = vth, W0 = W_blk, C0=C_blk, synapse='plastic')
    st = np.where(s_wp[0:n,:] != 0)
    spike_times_wp.append(st)
    W_blk_tot.append(W_blk)
    C_blk_tot.append(C_blk.copy())

    # --- structural plasticity: prune weak E->E and probabilistically grow new E->E
    if sp_structural:
        # calcium/activity proxy: average y_avg over the batch (per neuron)
        ca = np.mean(y_avg, axis=1)

        # pruning: remove weak existing excitatory connections
        prune_mask = (W_blk[0:ne, 0:ne] < prune_thresh) & (C_blk[0:ne, 0:ne] == 1)
        num_pruned = 0
        if np.any(prune_mask):
            num_pruned = int(np.sum(prune_mask))
            C_blk[0:ne, 0:ne][prune_mask] = 0.0
            W_blk[0:ne, 0:ne][prune_mask] = 0.0

        # growth: consider absent E->E synapses and grow based on presynaptic activity
        absent_rows, absent_cols = np.where(C_blk[0:ne, 0:ne] == 0)
        if len(absent_rows) > 0:
            # growth probability per candidate depends on presynaptic activity (row index)
            # row index = presynaptic neuron, column index = postsynaptic neuron
            def _sig(x):
                return 1.0 / (1.0 + np.exp(-grow_sig_slope * x))

            p_grow = p_sp * _sig(ca[absent_rows] - theta_grow)
            rand_vals = np.random.rand(len(p_grow))
            grow_idx = np.where(rand_vals < p_grow)[0]
            if len(grow_idx) > 0:
                rows = absent_rows[grow_idx]
                cols = absent_cols[grow_idx]
                # avoid self-connections
                keep = rows != cols
                rows = rows[keep]
                cols = cols[keep]
                C_blk[rows, cols] = 1.0
                # assign initial excitatory weight same as initial network (J)
                W_blk[rows, cols] = J
                num_grown = int(len(rows))
            else:
                num_grown = 0
        else:
            num_grown = 0

        # report structural changes this epoch
        total_conn = int(np.sum(C_blk[0:ne, 0:ne]))
        pruned_counts.append(num_pruned)
        grown_counts.append(num_grown)
        total_conn_counts.append(total_conn)
        print(f"Epoch {epoch}: pruned={num_pruned}, grown={num_grown}, E->E total={total_conn}")
    else:
        # structural plasticity disabled
        pruned_counts.append(0)
        grown_counts.append(0)
        total_conn_counts.append(int(np.sum(C_blk[0:ne, 0:ne])))
        print(f"Epoch {epoch}: structural plasticity disabled")
Wf = W_blk

#################################################################################
# -- after learning (Testing phase)
print('### after plasticity')

v0 = np.zeros((1,n)) 
# test using final connectivity mask
y_ap, s_ap, ym_plst_ap, yp_plst_ap, y_avg_ap, Wf_ap, C_final = \
    simulate_network(A = A, v0 = v0, x = x_ap, vth = vth, W0 = Wf, C0=C_blk, synapse='static')
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
    results['C0'] = C0

    results['spike_times_bp'] = spike_times_bp
    results['spike_times_wp'] = spike_times_wp
    results['spike_times_ap'] = spike_times_ap
    results['W_blk_tot'] = W_blk_tot
    results['C_blk_tot'] = C_blk_tot
    results['pruned_counts'] = pruned_counts
    results['grown_counts'] = grown_counts
    results['total_conn_counts'] = total_conn_counts
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
