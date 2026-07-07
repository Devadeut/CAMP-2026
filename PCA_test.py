##################################################################################
# pca_trajectories.py -- Projects population spike trains onto the top 3
# principal components and plots the resulting low-dimensional trajectories,
# in the style of trial-averaged PCA/state-space trajectory plots.
#
# Run this AFTER network_run.py has produced a 'results' pickle file.
##################################################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
import pickle
import params; from params import *
from network_simulator import simulate_network



# ---------------------------------------------------------------------------
# 1. Load results
# ---------------------------------------------------------------------------
with open('results', 'rb') as fl:
    results = pickle.load(fl)

spk_bp = results['spike_times_bp']          # before plasticity: (neuron_idx, time_idx)
spk_wp_tot = results['spike_times_wp']      # list, one entry per training batch
spk_ap = results['spike_times_ap']          # after plasticity

n_neurons = n                                # from params.py
T_bins = int(T)                              # time bins per batch/trial (dt=1ms)


# ---------------------------------------------------------------------------
# 2. Helpers: sparse (neuron_idx, time_idx) spike tuples -> smoothed rate matrix
# ---------------------------------------------------------------------------
def spikes_to_raster(spike_times, n_neurons, T_bins):
    """(neuron_idx, time_idx) tuple of arrays -> dense (n_neurons, T_bins) binary raster."""
    raster = np.zeros((n_neurons, T_bins))
    neuron_idx, time_idx = spike_times
    valid = time_idx < T_bins
    raster[neuron_idx[valid], time_idx[valid]] = 1
    return raster


def gaussian_kernel(sigma_ms, dt_=1):
    """Symmetric Gaussian smoothing kernel, +/- 3 sigma wide."""
    half_width = int(3 * sigma_ms / dt_)
    tt = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-tt**2 / (2. * sigma_ms**2))
    return kernel / kernel.sum()


def smooth_rate(raster, sigma_ms=20, dt_=1):
    """(n_neurons, T_bins) binary spikes -> (n_neurons, T_bins) smoothed rate (Hz)."""
    kernel = gaussian_kernel(sigma_ms, dt_)
    rate = np.array([np.convolve(row, kernel, mode='same') for row in raster])
    return rate * (1000. / dt_)   # spikes/bin -> Hz


def condition_rate_matrix(spike_times, sigma_ms=20):
    """(neuron_idx, time_idx) -> (T_bins, n_neurons) smoothed rate, time along rows."""
    raster = spikes_to_raster(spike_times, n_neurons, T_bins)
    rate = smooth_rate(raster, sigma_ms=sigma_ms, dt_=dt)
    return rate.T

# ---------------------------------------------------------------------------
# Baseline spontaneous (no stimulus) trajectories: run the network with
# untuned Poisson input using the saved initial and final weights.
# ---------------------------------------------------------------------------
W0 = results.get('W0')
C0 = results.get('C0', None)
Wf = None
Cf = None
if 'W_blk_tot' in results and len(results['W_blk_tot']) > 0:
    Wf = results['W_blk_tot'][-1]
if 'C_blk_tot' in results and len(results['C_blk_tot']) > 0:
    Cf = results['C_blk_tot'][-1]

def make_untuned_input(n_neurons, T_bins, rng=None):
    if rng is None:
        rng = np.random
    lam = (b_rate) * dt / 1000.0
    return rng.poisson(lam, (n_neurons, T_bins))

v0 = np.zeros((1, n_neurons))
# before plasticity spontaneous
spike_train_sp_before = make_untuned_input(n_neurons, T_bins, rng=np.random.RandomState(0))
Vb, Sb, _, _, _, _, _ = simulate_network(A, v0, spike_train_sp_before, vth, W0, C0, synapse='static')
spk_baseline_before = np.where(Sb != 0)

# after plasticity spontaneous (if final weights available)
if Wf is not None:
    spike_train_sp_after = make_untuned_input(n_neurons, T_bins, rng=np.random.RandomState(1))
    Vf, Sf, _, _, _, _, _ = simulate_network(A, v0, spike_train_sp_after, vth, Wf, Cf, synapse='static')
    spk_baseline_after = np.where(Sf != 0)
else:
    spk_baseline_after = (np.array([], dtype=int), np.array([], dtype=int))

conditions['Baseline before (no stimulus)'] = spk_baseline_before
conditions['Baseline after (no stimulus)'] = spk_baseline_after


# ---------------------------------------------------------------------------
# 3. Choose which conditions to compare, build ONE shared PCA space
#    (fit jointly on the concatenated data so trajectories are comparable)
# ---------------------------------------------------------------------------
conditions = {
    'Before plasticity':          spk_bp,
    'Early learning (batch 1)':   spk_wp_tot[0],
    'Late learning (final batch)': spk_wp_tot[-1],
    'After plasticity':           spk_ap,
}

# ---------------------------------------------------------------------------
# Baseline spontaneous (no stimulus) trajectories: run the network with
# untuned Poisson input using the saved initial and final weights.
# ---------------------------------------------------------------------------
W0 = results.get('W0')
C0 = results.get('C0', None)
Wf = None
Cf = None
if 'W_blk_tot' in results and len(results['W_blk_tot']) > 0:
    Wf = results['W_blk_tot'][-1]
if 'C_blk_tot' in results and len(results['C_blk_tot']) > 0:
    Cf = results['C_blk_tot'][-1]

def make_untuned_input(n_neurons, T_bins, rng=None):
    if rng is None:
        rng = np.random
    lam = (b_rate) * dt / 1000.0
    return rng.poisson(lam, (n_neurons, T_bins))

v0 = np.zeros((1, n_neurons))
# before plasticity spontaneous
spike_train_sp_before = make_untuned_input(n_neurons, T_bins, rng=np.random.RandomState(0))
Vb, Sb, _, _, _, _, _ = simulate_network(A, v0, spike_train_sp_before, vth, W0, C0, synapse='static')
spk_baseline_before = np.where(Sb != 0)

# after plasticity spontaneous (if final weights available)
if Wf is not None:
    spike_train_sp_after = make_untuned_input(n_neurons, T_bins, rng=np.random.RandomState(1))
    Vf, Sf, _, _, _, _, _ = simulate_network(A, v0, spike_train_sp_after, vth, Wf, Cf, synapse='static')
    spk_baseline_after = np.where(Sf != 0)
else:
    spk_baseline_after = (np.array([], dtype=int), np.array([], dtype=int))

conditions['Baseline before (no stimulus)'] = spk_baseline_before
conditions['Baseline after (no stimulus)'] = spk_baseline_after

rate_mats = {label: condition_rate_matrix(spk) for label, spk in conditions.items()}

all_rates = np.concatenate(list(rate_mats.values()), axis=0)   # (n_cond*T_bins, n_neurons)
mean_rate = all_rates.mean(axis=0, keepdims=True)
centered = all_rates - mean_rate

# PCA via SVD -- avoids an sklearn dependency
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
n_components = 3
components = Vt[:n_components]                        # (3, n_neurons)
explained_var = (S**2 / np.sum(S**2))[:n_components]
print('Variance explained by PC1-3:', np.round(explained_var, 3),
      '(cumulative: %.1f%%)' % (100 * explained_var.sum()))


def project(rate_matrix):
    return (rate_matrix - mean_rate) @ components.T    # (T_bins, 3)


projections = {label: project(mat) for label, mat in rate_mats.items()}


# ---------------------------------------------------------------------------
# 4. Plot: 3D trajectories with start (square) / end (circle) markers
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.viridis(np.linspace(0, 1, len(projections)))

for (label, traj), color in zip(projections.items(), colors):
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, lw=1.5, alpha=.85, label=label)
    ax.scatter(*traj[0],  color=color, marker='s', s=80, edgecolor='k', zorder=5)  # start
    ax.scatter(*traj[-1], color=color, marker='o', s=80, edgecolor='k', zorder=5)  # end

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Population activity trajectories in shared PC space\n'
              '(square = trial start, circle = trial end)')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)

plt.tight_layout()
plt.savefig('Fig-PCA-trajectories.png', dpi=200, bbox_inches='tight')
plt.show()

# ----