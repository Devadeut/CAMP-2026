##################################################################################
# plot_figures.py -- Reads and analyzes the results generated from network_run.py
# and plots Figures 1 and 3 in:
#
# Ref: Sadeh, Clopath and Rotter (PLOS Computational Biology, 2015).
# Emergence of Functional Specificity in Balanced Networks with Synaptic Plasticity.
#
#
##################################################################################
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload  # Python 3 compatibility
import params; reload(params); from params import *
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable

################################################################################
# --- read the results
with open('results', 'rb') as fl:
    results = pickle.load(fl)
W0 = results['W0']
spk_bp = results['spike_times_bp']
spk_wp_tot = results['spike_times_wp']
spk_ap = results['spike_times_ap']
spk_sp_tot = results['spk_sp_tot']
spk_cd_tot = results['spk_cd_tot']
W_blk_tot = results['W_blk_tot']
stim_rng_tot = results['stim_rng_tot']
W_sp_tot = results['W_sp_tot']
W_cd_tot = results['W_cd_tot']
stim_rng_tot_cd = results['stim_rng_tot_cd']
Wf = W_blk_tot[-1]

################################################################################
### Figure 1
#########
mksz = 2.
Ts = T / 1000
sim_time = int(T)


def _temp_plot_(spk, ax, stim=0, yy=0):
    exid = np.where(spk[0] < ne)[0]
    inid = np.where(spk[0] >= ne)[0]
    htex = np.histogram(spk[1][exid], bins=int(sim_time / 10), range=(0, sim_time))
    htin = np.histogram(spk[1][inid], bins=int(sim_time / 10), range=(0, sim_time))
    hr = np.histogram(spk[0], bins=n, range=(0, n))
    ax.plot(spk[1][exid] * dt, spk[0][exid], 'r.', markersize=mksz, label='Exc: ' + str(np.round(len(exid) / ne)))
    ax.plot(spk[1][inid] * dt, spk[0][inid], 'b.', markersize=mksz, label='Inh: ' + str(np.round(len(inid) / ne)))
    ax.set_yticks([0, 99, 199, 299, ne - 1, n - 1])
    ax.set_yticklabels([])
    ax.set_ylim(0 - 10, n + 10)
    ax.set_xlim([0 - 10, sim_time + 10])
    ax.set_xticklabels([])
    divider = make_axes_locatable(ax)
    axHisty = divider.append_axes("right", size=.5, pad=0.1)
    # adjust_spines(axHisty,['left', 'bottom'], outward=0)
    axHisty.plot(hr[0] / (Ts), hr[1][0:-1], color='k', lw=2)
    if stim == 0:
        plt.text(.85, .5, str(np.round(len(exid) / ne / Ts, 1)) + ' Hz', transform=axHisty.transAxes, color='r')
        plt.text(.85, .85, str(np.round(len(inid) / ni / Ts, 1)) + ' Hz', transform=axHisty.transAxes, color='b')
    else:
        plt.text(.9, .5, str(np.round(len(exid) / ne / Ts, 1)) + ' Hz', transform=axHisty.transAxes, color='r')
        plt.text(.9, .85, str(np.round(len(inid) / ni / Ts, 1)) + ' Hz', transform=axHisty.transAxes, color='b')
    axHisty.set_yticks([0, 99, 199, 299, ne - 1, n - 1])
    axHisty.set_yticklabels([])
    axHisty.set_xticks([0, 10])
    axHisty.set_ylim(0 - 10, n + 10)
    axHistx = divider.append_axes("bottom", 1.2, pad=0.3)
    # adjust_spines(axHistx,['left', 'bottom'], outward=0)
    axHistx.plot(htex[1][0:-1], htex[0], color='r', lw=2, label='Exc')
    axHistx.plot(htin[1][0:-1], htin[0], color='b', lw=2, label='Inh')
    axHistx.set_yticks([0, 50, 100, 150])
    axHistx.set_yticklabels([])
    if yy == 1:
        axHistx.set_xlabel('Time (ms)')
        axHistx.set_ylabel('Population spike count')
        axHistx.set_yticklabels([0, 50, 100, 150])
        plt.legend(loc=1, frameon=False, prop={'size': 12.5})
        axHisty.set_xlabel('Firing rate \n (spikes/s)', size=10)


fig = plt.figure(figsize=(16, 8))
mycl = plt.imshow(np.random.uniform(0, 1, (100, 100)), cmap='hsv', vmin=0, vmax=1)
plt.clf()
##
ax1 = plt.subplot(141)
plt.title('Before Plasticity')
_temp_plot_(spk=spk_bp, ax=ax1, yy=1)
ax1.set_ylabel('Neuron #')
# ax1.set_yticks([0, 99, 199, 299, ne, n])
ax1.set_yticklabels([1, 100, 200, 300, ne, n])
for i in range(int(stim_no)):
    ax1.plot([i * trial_time, (i + 1) * trial_time], [-5, -5], '-', color=plt.cm.hsv(th / np.pi), lw=10)
##
ax2 = plt.subplot(142)
plt.title('Beginning of Plasticity')
_temp_plot_(spk=spk_wp_tot[0], ax=ax2, stim=1)
for i in range(int(stim_no)):
    clbr = ax2.plot([i * trial_time, (i + 1) * trial_time], [-5, -5], '-',
                     color=plt.cm.hsv(stim_rng_tot[0][i] / np.pi), lw=10)
cax = fig.add_axes([.485, .25, .01, .1])
clbr = plt.colorbar(mycl, cax=cax, orientation='vertical')
clbr.set_ticks([0, .25, .5, .75, 1])
clbr.set_ticklabels([0, 45, 90, 135, 180])
##
ax3 = plt.subplot(143)
plt.title('End of Plasticity')
_temp_plot_(spk=spk_wp_tot[block_no - 1], ax=ax3, stim=2)
for i in range(int(stim_no)):
    ax3.plot([i * trial_time, (i + 1) * trial_time], [-5, -5], '-',
             color=plt.cm.hsv(stim_rng_tot[-1][i] / np.pi), lw=10)
##
ax4 = plt.subplot(144)
plt.title('After Plasticity')
_temp_plot_(spk=spk_ap, ax=ax4)
for i in range(int(stim_no)):
    ax4.plot([i * trial_time, (i + 1) * trial_time], [-5, -5], '-', color=plt.cm.hsv(th / np.pi), lw=10)
ax4.text(.25, .1, 'Sparser activity', size=15, transform=ax4.transAxes)
plt.subplots_adjust(left=.05, right=.95, bottom=.075, top=.95, wspace=.25)
plt.savefig('Fig1')

################################################################################
### Figure 3 (A-C)
###########
plt.figure(figsize=(14, 5))
plt.subplot(131)
plt.title('Initial Weights (W0)')
plt.imshow(W0)
clb = plt.colorbar(shrink=.75)
clb.set_ticks([-4, 0, .5])
clb.set_ticklabels([-4, 0, .5])
plt.xlabel('Post-synaptic #')
plt.ylabel('Pre-synaptic #')
plt.subplot(132)
plt.title('Final Weights (Wf)')
plt.imshow(Wf, vmin=-5, vmax=2)
clb = plt.colorbar(shrink=.75)
clb.set_ticks([-5, -4, -3, -2, -1, 0, 1, 2])
clb.set_ticklabels([-5, -4, -3, -2, -1, 0, 1, 2])
plt.subplot(133)
plt.title('Weight Changes (Wf - W0)')
plt.imshow(Wf - W0, vmin=-1.5, vmax=1.5)
clb = plt.colorbar(shrink=.75)
clb.set_ticks([-1.5, -1, -.5, 0, .5, 1, 1.5])
clb.set_ticklabels([-1.5, -1, -.5, 0, .5, 1, 1.5])
plt.savefig('Fig3A')
###
## bid and fs
dW = Wf - W0
dpo, dw_po, Wf_po, W0_po = [], [], [], []
for ii in range(ne):
    for jj in range(ne):
        Wf_po.append(Wf[ii, jj])
        W0_po.append(W0[ii, jj])
        dpo.append(po_init[ii] - po_init[jj])
W0_po, Wf_po = np.array(W0_po), np.array(Wf_po)
dw_po = np.array(dw_po)  # NOTE: dw_po is never populated in the original script either; kept as dead code, see note 5 above.
dpo = np.array(dpo)
dpo_id1 = np.where((abs(dpo) < np.pi / 6) | (abs(dpo) > np.pi - np.pi / 6))
dpo_id2 = np.where(
    ((np.abs(dpo) < 2 * np.pi / 6) & (np.abs(dpo) > np.pi / 6)) |
    ((np.abs(dpo) > np.pi - 2 * np.pi / 6) & (np.abs(dpo) < np.pi - np.pi / 6))
)
dpo_id3 = np.where(
    ((np.abs(dpo) < 3 * np.pi / 6) & (np.abs(dpo) > 2 * np.pi / 6)) |
    ((np.abs(dpo) > np.pi - 3 * np.pi / 6) & (np.abs(dpo) < np.pi - 2 * np.pi / 6))
)

################################################################################
### Figure 3 (D-G)
###########
plt.figure(figsize=(14, 5))


# plt.title('aligend weights')
def _plot_alignw_(rng1=range(0, ne), rng2=range(ne, n)):
    for i in rng1:
        ax.plot(-po_init[rng2] + po_init[i], Wf[i][rng2], 'k.', ms=mksz, alpha=.5)
    for i in rng1:
        if i == rng1[0]:
            ax.plot(-po_init[rng2] + po_init[i], W0[i][rng2], 'r.', ms=1, alpha=1.)
        else:
            ax.plot(-po_init[rng2] + po_init[i], W0[i][rng2], 'r.', ms=mksz, alpha=.5)


ax = plt.subplot(141)
plt.title('Exc to Exc')
_plot_alignw_(rng1=range(0, ne), rng2=range(0, ne))
ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_xticklabels(['', -90, 0, 90, ''])
ax.set_yticks([0, .5, 1, 1.5, 2])
ax.set_yticklabels([0, .5, 1, 1.5, 2])
ax.set_ylim([0 - .1, 2 + .1])
ax.set_ylabel('Final Weights (mV)')
ax.set_xlabel('Pre-syn. PO - Post-syn. PO (deg)')

ax = plt.subplot(142)
plt.title('Exc to Inh')
_plot_alignw_(rng1=range(0, ne), rng2=range(ne, n))
ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_xticklabels(['', -90, 0, 90, ''])
ax.set_yticks([0, .5, 1, 1.5, 2])
ax.set_yticklabels([0, .5, 1, 1.5, 2])
ax.set_ylim([0 - .1, 2 + .1])
ax.set_ylabel('Final Weights (mV)')

ax = plt.subplot(143)
plt.title('Inh to Exc')
_plot_alignw_(rng1=range(ne, n), rng2=range(0, ne))
ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_xticklabels(['', -90, 0, 90, ''])
ax.set_yticks([-6, -5.5, -5, -4.5, -4, -3.5, -3])
ax.set_yticklabels([-6, -5.5, -5, -4.5, -4, -3.5, -3])
ax.set_ylim([-5.5 - .1, -3 + .1])
ax.set_ylabel('Final Weights (mV)')

ax = plt.subplot(144)
# adjust_spines(ax,['left', 'bottom'], outward=0, s=0)
plt.title('Connection Specificity')
xrng = [1, 2, 3]


def _part_dpo_(www, cl='k', lbl=[]):
    ym1, ys1 = np.mean(www[dpo_id1]), np.std(www[dpo_id1])
    ym2, ys2 = np.mean(www[dpo_id2]), np.std(www[dpo_id2])
    ym3, ys3 = np.mean(www[dpo_id3]), np.std(www[dpo_id3])
    ymrng = np.array([ym1, ym2, ym3])
    ysrng = np.array([ys1, ys2, ys3])
    ax.plot(xrng, ymrng, '-o', lw=2, color=cl, label=lbl)


_part_dpo_(W0_po, cl='r', lbl='Initial')
_part_dpo_(Wf_po, lbl='Final')
plt.legend(title='Exc to Exc', frameon=False, numpoints=1)
ax.set_xlim(0, 4)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['0-30', '30-60', '60-90'], rotation=0)
ax.set_yticks([0, .2, .4, .6, .8, 1])
ax.set_yticklabels([0, .2, .4, .6, .8, 1])
ax.set_ylim(0, 1)
plt.xlabel('dpo range (deg)')
plt.ylabel('Average Weight (mV)')
plt.subplots_adjust(left=.05, right=.97, bottom=.15, top=.925, wspace=.45)
plt.savefig('Fig3B')
plt.show()
# ----------

# plot to show the numbe rof synapses pruned and grown over time and total number of connections over time
pruned_counts = results['pruned_counts']
grown_counts = results['grown_counts']
total_conn_counts = results['total_conn_counts']
plt.figure(figsize=(14, 5))
plt.subplot(131)
plt.title('Number of Synapses Pruned')
plt.plot(pruned_counts, 'k', lw=2)
plt.xlabel('Block #')
plt.ylabel('Number of Synapses Pruned')
plt.subplot(132)
plt.title('Number of Synapses Grown')
plt.plot(grown_counts, 'k', lw=2)
plt.xlabel('Block #')
plt.ylabel('Number of Synapses Grown')
plt.subplot(133)
plt.title('Total Number of Connections')
plt.plot(total_conn_counts, 'k', lw=2)
plt.xlabel('Block #')
plt.ylabel('Total Number of Connections')
plt.tight_layout()
plt.savefig('Fig-structural-plasticity')
plt.show()

