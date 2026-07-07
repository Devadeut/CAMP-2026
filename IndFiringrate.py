#################################################################################
# -- spontaneous firing rate (no orientation tuning)

print('### spontaneous firing rate')

T = sim_time                     # simulation time (ms)
bins = int(T / dt)

# Constant untuned Poisson input
spike_train_sp = np.zeros((n, bins))

for i in range(n):
    spike_train_sp[i, :] = np.random.poisson(b_rate * dt / 1000., bins)

# Initial membrane potential
v0 = np.zeros((1, n))

# Run network without plasticity
y_sp, s_sp, ym_sp, yp_sp, y_avg_sp, W_sp = simulate_network(
    A=A,
    v0=v0,
    x=spike_train_sp,
    vth=vth,
    W0=W0,
    synapse='static'
)

# Compute firing rate of each neuron (Hz)
T_sec = T / 1000.0
firing_rates = np.sum(s_sp, axis=1) / T_sec

# Population statistics
print(f"Mean excitatory firing rate: {np.mean(firing_rates[:ne]):.2f} Hz")
print(f"Mean inhibitory firing rate: {np.mean(firing_rates[ne:]):.2f} Hz")

# firing_rates[i] gives the spontaneous firing rate of neuron i
