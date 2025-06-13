
# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# Optimization of atomic clusters using Pulser and QuTiP
# Author : PAYA Ronan
# Contributor : ...
# --------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qutip
import os
import pickle

from pulser import Register, Pulse, Sequence
from pulser.devices import AnalogDevice
from pulser.waveforms import BlackmanWaveform, ConstantWaveform
from pulser_simulation import QutipEmulator
from qutip import basis, tensor, qeye, fidelity
from itertools import combinations
import traceback


# Settings
n_clusters = 4
n_qubits = 5
weights = np.array([1, 1, 1, 1])
g_proj = basis(2, 0).proj()
id_qubit = qeye(2)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
SHOW_PLOTS = False  # Toggle to True to display plots during optimization

scores_history = []  # History of best scores

# Functions
def scale_and_center(register: Register, scale: float = 1.0) -> Register:
    xs, ys = zip(*register.qubits.values())
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return Register({k: ((x - cx) * scale, (y - cy) * scale)
                     for k, (x, y) in register.qubits.items()})

def add_random_offset(register: Register, max_offset: float = 1.0) -> Register:
    return Register({k: (x + np.random.uniform(-max_offset, max_offset),
                        y + np.random.uniform(-max_offset, max_offset))
                        for k, (x, y) in register.qubits.items()})

# Safe base spacing (>= 5 µm required)
spacing = 6.0

base_qubits = {
    "q1": (1.8922982034770317, -8.724927619527081),
    "q2": (7.2123009500590625, -6.384919684956765),
    "q3": (11.385628586777813, -7.924925330708719),
    "q4": (2.81230018711961, -13.118258155415754),
    "q5": (6.34563224888718750, -10.531589759419658),
}

def generate_cluster_centers(n_clusters, min_sep=15.0, max_range=30.0):
    centers = []
    attempts = 0
    while len(centers) < n_clusters:
        cx, cy = np.random.uniform(-max_range, max_range, size=2)
        if all(np.linalg.norm(np.array([cx, cy]) - np.array(c)) >= min_sep for c in centers):
            centers.append((cx, cy))
        attempts += 1
        if attempts > 500:
            raise RuntimeError("Unable to generate well-separated centers.")
    return centers

def generate_cluster(center, n_qubits, radius, min_dist):
    qubits = []
    attempts = 0
    while len(qubits) < n_qubits:
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)

        too_close = any(np.hypot(x - x2, y - y2) < min_dist for (x2, y2) in qubits)
        if not too_close:
            qubits.append((x, y))

        attempts += 1
        if attempts > 1000:
            raise RuntimeError("Unable to place all qubits without collision in the cluster.")
    return qubits

def evaluate_configuration():
    cluster_centers = generate_cluster_centers(n_clusters=4)
    registers = []
    for i, center in enumerate(cluster_centers):
        try:
            positions = generate_cluster(center, n_qubits=5, radius=6.0, min_dist=5.0)
            qubits = {f"c{i+1}_q{j+1}": (x, y) for j, (x, y) in enumerate(positions)}
            reg = Register(qubits)
            AnalogDevice.validate_register(reg)
            registers.append(reg)
        except Exception as e:
            continue

    if not registers:
        raise RuntimeError("No valid cluster generated. Restart required.")

    for qname, (x, y) in qubits.items():
        if np.hypot(x, y) > 38.0:
            raise ValueError(f"{qname} too far from the global center : {np.hypot(x, y):.2f} µm")

    registers = sorted(registers, key=lambda r: sorted(r.qubits.keys())[0])


    amp1 = BlackmanWaveform(4000, 15.30648)
    composite = ConstantWaveform(2000)
    composite_1 = ConstantWaveform(2000)
    det1 = CompositeWaveform(composite, composite_1)
    pulse1 = Pulse(amp1, det1, 3.4871678454846706, 6.283185307179586)

    composite_2 = BlackmanWaveform_1.from_max_val(0, 0)
    composite_3 = ConstantWaveform(2000)
    amp2 = CompositeWaveform(composite_2, composite_3)
    pulse2 = Pulse.ConstantDetuning(amp2, 8.482, 3.4871678454846706)

    

    sequences = []
    for i, reg in enumerate(registers):
        try:
            seq = Sequence(reg, AnalogDevice)
            seq.declare_channel(f"ch{i+1}", "rydberg_global")
            seq.add(pulse1, "ch2")
            seq.add(pulse2, "ch2")
            sequences.append(seq)
        except Exception:
            continue

    if not sequences:
        raise RuntimeError("No valid sequence generated. Stopping.")

    sim_results = [QutipEmulator.from_sequence(seq, sampling_rate=0.05).run()
                for seq in sequences]

    n_atoms = len(sequences[0].register.qubits)
    psi_target = tensor([basis(2, 1) for _ in range(n_atoms)])

    fidelity_list = [[fidelity(state, psi_target) for state in res.states]
                    for res in sim_results]

    obs_all = sum(tensor([g_proj if i == j else id_qubit for i in range(n_atoms)])
                for j in range(n_atoms)) / n_atoms
    expectations = [res.expect([obs_all])[0] for res in sim_results]
    times = sim_results[0]._sim_times

    if len(weights) != len(expectations):
        mean_exp = np.mean(expectations, axis=0)
        std_exp = np.std(expectations, axis=0)
        mean_fidelity = np.mean(fidelity_list, axis=0)
    else:
        mean_exp = np.average(expectations, axis=0, weights=weights)
        std_exp = np.sqrt(np.average((expectations - mean_exp) ** 2, axis=0, weights=weights))
        mean_fidelity = np.average(fidelity_list, axis=0, weights=weights)

    final_score = mean_fidelity[-1]
    return final_score, expectations, fidelity_list, registers, times, mean_exp, std_exp, mean_fidelity


if os.path.exists("best_config.pkl"):
    with open("best_config.pkl", "rb") as f:
        best_data = pickle.load(f)
    best_score = best_data[0]
    print(f"Existing configuration loaded. Fidelity: {best_score:.4f}")
else:
    best_score = -1
    best_data = None
    print("No existing configuration. Starting a new search.")

n_rounds = 10 #Number of global round
n_trials_per_round = 20 #Number of try per global round

for round_id in range(1, n_rounds + 1):
    print(f"\n=== ROUND {round_id} ===")
    improved = False
    for trial in range(1, n_trials_per_round + 1):
        try:
            result = evaluate_configuration()
            score = result[0]
            if score > best_score:
                best_score = score
                best_data = result
                improved = True
                scores_history.append((round_id, trial, score))
                print(f"New better loyalty : {score:.4f} (round {round_id}, trial {trial})")
        except Exception as e:
            continue
    if not improved:
        print(f"No improvement during the round {round_id}.")
    else:
        with open("best_config.pkl", "wb") as f:
            pickle.dump(best_data, f)
        print("Saved configuration (improved).")

from datetime import datetime

# add date For all new session 
with open("historical_scores.txt", "a") as f:
    f.write(f"\n--- Session of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    for r, t, s in scores_history:
        f.write(f"Round {r}, Trial {t}, Score: {s:.5f}\n")
    
    print(f"\nMaximum final fidelity achieved : {best_score:.4f}")

output_dir = "best_run_outputs"
os.makedirs(output_dir, exist_ok=True)

if best_data:
    score, expectations, fidelity_list, registers, times, mean_exp, std_exp, mean_fidelity = best_data
else:
    raise RuntimeError("No best_data available to plot results.")

# Average population per cluster
plt.figure(figsize=(8, 5))
for i, exp in enumerate(expectations):
    plt.plot(times, exp, label=f"Cluster {i+1}", color=colors[i])
plt.plot(times, mean_exp, color='black', linewidth=2, label="Weighted average")
plt.fill_between(times, mean_exp - std_exp, mean_exp + std_exp, alpha=0.2, color='gray')
plt.xlabel("Time (µs)")
plt.ylabel("Population ( $\\vert g \\rangle$ )")
plt.title("Average population observed per cluster")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "population_per_cluster.png"), dpi=300)
plt.close()

# Temporal fidelity
plt.figure(figsize=(8, 4))
for i, fid in enumerate(fidelity_list):
    plt.plot(times, fid, label=f"Cluster {i+1}", color=colors[i])
plt.plot(times, mean_fidelity, color='black', linewidth=2, label="Weighted average fidelity")
plt.xlabel("Temps (µs)")
plt.ylabel("fidelity vs $|r...r⟩$")
plt.title("Temporal fidelity of clusters")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fidelite_per_cluster.png"), dpi=300)
plt.close()

# Final fidelity Heatmap
fidelity_final = [fidel[-1] for fidel in fidelity_list]
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 1.5))
sns.heatmap([fidelity_final], annot=True, fmt=".3f", cmap="viridis",
            xticklabels=[f"C{i+1}" for i in range(n_clusters)], yticklabels=["Fin"])
plt.title("Final fidelity by cluster (t = end)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmap_fidelity_finale.png"), dpi=300)
plt.close()

# Spatial distribution
plt.figure(figsize=(7, 7))
for i, reg in enumerate(registers):
    coords = list(reg.qubits.values())
    xs = [float(x) for x, _ in coords]
    ys = [float(y) for _, y in coords]
    labels = list(reg.qubits.keys())
    plt.scatter(xs, ys, label=f"Cluster {i+1}", color=colors[i], s=60, alpha=0.8)
    for x, y, label in zip(xs, ys, labels):
        plt.text(float(x) + 0.3, float(y) + 0.3, label, fontsize=7, color=colors[i])

plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Spatial distribution of qubits per cluster")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spatial_map_clusters.png"), dpi=300)
plt.close()

print(f"Charts saved in : {output_dir}/")