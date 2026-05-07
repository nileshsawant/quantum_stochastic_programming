"""
Parallelisation report: 1-GPU vs 2-GPU (single-node mqpu) vs 4-GPU (multi-node mpi4py).
Data from par_benchmark.py and par_benchmark_mpi.py on Kestrel H100s, 256 shots.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRATCH_IMPL = '/kfs3/scratch/nsawant/quantum_stochastic_programming/qiskit_impl'

# Timings: ny -> (t_1gpu, t_2gpu, t_4gpu)  -- None = not measured
# 1-GPU and 2-GPU: nvidia mqpu (single node), 256 shots
# 4-GPU: mpi4py shot splitting across 2 nodes x 2 GPUs, 256 shots total
DATA = {
    4:  (2.2,   2.6,   None),
    6:  (3.9,   5.4,   None),
    8:  (6.5,  10.4,   None),
    10: (46.9,  24.9,  12.0),
    12: (69.2,  36.1,  19.3),
    14: (808.1, 393.7, 70.0),
}
N_SHOTS = 256

NY_ALL  = sorted(DATA)                      # all n_y for 1-GPU vs 2-GPU
NY_MPI  = [ny for ny in NY_ALL if DATA[ny][2] is not None]  # 4-GPU subset

t1_all = np.array([DATA[ny][0] for ny in NY_ALL])
t2_all = np.array([DATA[ny][1] for ny in NY_ALL])
su2_all = t1_all / t2_all
eff2_all = su2_all / 2 * 100

t1_mpi = np.array([DATA[ny][0] for ny in NY_MPI])
t2_mpi = np.array([DATA[ny][1] for ny in NY_MPI])
t4_mpi = np.array([DATA[ny][2] for ny in NY_MPI])
su2_mpi = t1_mpi / t2_mpi
su4_mpi = t1_mpi / t4_mpi
eff2_mpi = su2_mpi / 2 * 100
eff4_mpi = su4_mpi / 4 * 100

# ── Figure: 3 panels ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── Panel 1: Full wall-time comparison (log scale) ────────────────────────────
ax = axes[0]
xa = np.arange(len(NY_ALL))
bw = 0.27
b1 = ax.bar(xa - bw,   t1_all, width=bw, color='steelblue', alpha=0.85, label='1 GPU (baseline)')
b2 = ax.bar(xa,        t2_all, width=bw, color='tomato',    alpha=0.85, label='2 GPU mqpu (single-node)')

# 4-GPU bars only for measured n_y
for i, ny in enumerate(NY_ALL):
    t4 = DATA[ny][2]
    if t4 is not None:
        ax.bar(xa[i] + bw, t4, width=bw, color='seagreen', alpha=0.85,
               label='4 GPU mpi4py (multi-node)' if i == NY_ALL.index(NY_MPI[0]) else '')

for rect, v in zip(list(b1)+list(b2), list(t1_all)+list(t2_all)):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()*1.06,
            f'{v:.0f}s' if v >= 10 else f'{v:.1f}s',
            ha='center', va='bottom', fontsize=6.5)
for i, ny in enumerate(NY_ALL):
    t4 = DATA[ny][2]
    if t4 is not None:
        xpos = xa[i] + bw
        ax.text(xpos, t4*1.06, f'{t4:.0f}s', ha='center', va='bottom', fontsize=6.5)

ax.set_yscale('log'); ax.set_ylim(bottom=0.5)
ax.set_xticks(xa)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY_ALL], fontsize=9)
ax.set_ylabel('Wall time (s)', fontsize=11)
ax.set_title(f'Noisy sampling wall time\n({N_SHOTS} shots, TIMESTEPS=20)', fontsize=11)
ax.legend(fontsize=8, loc='upper left'); ax.grid(True, axis='y', alpha=0.4)

# ── Panel 2: Speedup (1-GPU vs 2-GPU and 4-GPU, for MPI sizes) ───────────────
ax = axes[1]
xm = np.arange(len(NY_MPI))
bw2 = 0.35
ax.bar(xm - bw2/2, su2_mpi, width=bw2, color='tomato',   alpha=0.85, label='2-GPU speedup')
ax.bar(xm + bw2/2, su4_mpi, width=bw2, color='seagreen', alpha=0.85, label='4-GPU speedup')
for i, (s2, s4) in enumerate(zip(su2_mpi, su4_mpi)):
    ax.text(xm[i] - bw2/2, s2+0.05, f'{s2:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(xm[i] + bw2/2, s4+0.05, f'{s4:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(2, color='tomato',   lw=1.0, ls=':', alpha=0.6, label='Ideal 2x')
ax.axhline(4, color='seagreen', lw=1.0, ls=':', alpha=0.6, label='Ideal 4x')
ax.set_xticks(xm)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY_MPI], fontsize=10)
ax.set_ylabel('Speedup  T_1GPU / T_nGPU', fontsize=11)
ax.set_title('2-GPU vs 4-GPU speedup\n(vs 1-GPU baseline)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)
ax.set_ylim(0, max(su4_mpi) * 1.25)

# ── Panel 3: Parallel efficiency ──────────────────────────────────────────────
ax = axes[2]
ax.bar(xm - bw2/2, eff2_mpi, width=bw2, color='tomato',   alpha=0.85, label='2-GPU eff.')
ax.bar(xm + bw2/2, eff4_mpi, width=bw2, color='seagreen', alpha=0.85, label='4-GPU eff.')
for i, (e2, e4) in enumerate(zip(eff2_mpi, eff4_mpi)):
    ax.text(xm[i] - bw2/2, e2+1, f'{e2:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(xm[i] + bw2/2, e4+1, f'{e4:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(100, color='grey', lw=0.8, ls=':', label='Ideal 100%')
ax.axhline(80,  color='gold', lw=0.8, ls='--', alpha=0.7, label='80% threshold')
ax.set_xticks(xm)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY_MPI], fontsize=10)
ax.set_ylabel('Parallel efficiency (%)', fontsize=11)
ax.set_title('Parallel efficiency\n(multi-node n_y = 10..14)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)
ax.set_ylim(0, max(eff4_mpi) * 1.25)

fig.suptitle(
    'CUDA-Q Shot-Splitting Parallelisation  |  '
    f'{N_SHOTS} shots  |  p1=0.0001, p2=0.001  |  H100 (Kestrel)',
    fontsize=11, y=1.02)

plt.tight_layout()
out = os.path.join(SCRATCH_IMPL, 'parallelisation_study.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Plot saved -> {out}')

# Summary table
print(f"\n{'n_y':>4}  {'qubits':>6}  {'1-GPU(s)':>9}  {'2-GPU(s)':>9}  "
      f"{'4-GPU(s)':>9}  {'Su_2x':>7}  {'Su_4x':>7}  {'Eff_2x':>7}  {'Eff_4x':>7}")
print('-' * 85)
for ny in NY_ALL:
    t1s, t2s, t4s = DATA[ny]
    su2 = t1s/t2s
    e2  = su2/2*100
    if t4s is not None:
        su4 = t1s/t4s
        e4  = su4/4*100
        print(f'{ny:>4}  {2*ny:>6}  {t1s:>9.1f}  {t2s:>9.1f}  '
              f'{t4s:>9.1f}  {su2:>7.2f}x  {su4:>7.2f}x  {e2:>6.0f}%  {e4:>6.0f}%')
    else:
        print(f'{ny:>4}  {2*ny:>6}  {t1s:>9.1f}  {t2s:>9.1f}  '
              f'{"—":>9}  {su2:>7.2f}x  {"—":>7}  {e2:>6.0f}%  {"—":>6}')
