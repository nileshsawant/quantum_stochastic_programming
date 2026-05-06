"""
Noise study: DQA accuracy (ideal vs depolarizing) for n_y in {4, 6, 8, 10}.

Key design choices
------------------
* COBYLA uses estimate_expected_value_observe() — a single cudaq.observe() GPU
  call that computes <psi|H|psi> as a tensor contraction.  No Python loop over
  2^(2*n_y) states — O(1) GPU calls vs O(2^(2*n_y)) Python iterations.
* P_LAYERS=2 (p=2 DQA layers, 4 parameters) — COBYLA-friendly regardless of n_y.
* Reports both error-vs-classical and pure noise degradation.

Run with:
  /kfs3/scratch/nsawant/qiskit_env/bin/python run_noise_study.py 2>&1 | tee /tmp/noise_study.log
"""
import os, sys, math, time, json

_QISKIT_SP   = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_QISKIT_IMPL = '/kfs3/scratch/nsawant/quantum_stochastic_programming/qiskit_impl'
for _p in [_QISKIT_SP, _QISKIT_IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_QISKIT_IMPL)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import cudaq
from cudaq_impl import CudaqQAEOptimizer, build_depolarizing_noise_model

cudaq.set_target('nvidia')
print(f'CUDA-Q  target : {cudaq.get_target().name}')
print(f'GPUs available : {cudaq.num_available_gpus()}')

# ── NOISE MODEL ───────────────────────────────────────────────────────────────
P1, P2  = 0.001, 0.010
N_TRAJ  = 2048
noise_model = build_depolarizing_noise_model(p1=P1, p2=P2)
print(f'Noise model    : p1={P1}, p2={P2}, trajectories={N_TRAJ}\n')

# ── PROBLEM CONFIGS ───────────────────────────────────────────────────────────
_BASE_CY = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
N_Y_LIST = [4, 6, 8, 10]
P_LAYERS = 2       # fixed DQA depth: 4 parameters always
C_R      = 10.0

CONFIGS = {}
for ny in N_Y_LIST:
    w  = ny // 2
    cy = _BASE_CY[:ny]
    cost_norm = w * C_R / ny
    # adiabatic warm-start angles
    theta0 = []
    for t in range(P_LAYERS):
        s = (t + 1) / P_LAYERS
        theta0 += [s * math.pi / 4, (1 - s) * math.pi / 4]
    CONFIGS[ny] = dict(n_y=ny, w_d=w, c_y=cy, c_r=C_R,
                       cost_norm=cost_norm, theta0=theta0)
    print(f'  n_y={ny:2d}  w_d={w}  cost_norm={cost_norm:.2f}  '
          f'#params={len(theta0)}  c_y={[round(c,2) for c in cy]}')

# ── CLASSICAL REFERENCE ─────────────────────────────────────────────────────
# Brute-force E[Q] with optimal y* (commit w cheapest turbines) over uniform xi
print('\n── Classical brute-force E[Q] ──')
classical_phi = {}
for ny, cfg in CONFIGS.items():
    cy, cr, w = cfg['c_y'], cfg['c_r'], cfg['w_d']
    y_star = [0] * ny
    for j in sorted(range(ny), key=lambda j: cy[j])[:w]:
        y_star[j] = 1
    total = 0.0
    for xi_int in range(2 ** ny):
        for j in range(ny):
            if y_star[j]:
                total += cy[j] if (xi_int >> j) & 1 else cr
    classical_phi[ny] = total / (2 ** ny)
    print(f'  n_y={ny}  y*={y_star}  φ_classical={classical_phi[ny]:.4f}')

# ── COBYLA WITH cudaq.observe() — O(1) GPU calls per iteration ───────────────
# estimate_expected_value_observe uses cudaq.observe() to compute <psi|H|psi>
# as a single tensor contraction on the GPU.  No Python loop over 2^(2*n_y)
# states.  For n_y=10 this is ~10000x faster than estimate_expected_value_sv.
print('\n── COBYLA optimisation using cudaq.observe() ──')
optimized_thetas = {}
cobyla_phi = {}
for ny, cfg in CONFIGS.items():
    print(f'\n  n_y={ny}  ({len(cfg["theta0"])} params)...', flush=True)
    opt = CudaqQAEOptimizer(
        c_x=[3.]*ny, c_y=cfg['c_y'], c_r=cfg['c_r'],
        n_y=ny, w_d=cfg['w_d'], cost_norm=cfg['cost_norm'],
        noise_model=None,
    )
    calls = [0]
    def _obj(th, _opt=opt):
        calls[0] += 1
        return _opt.estimate_expected_value_observe(th.tolist())  # single GPU call

    t0  = time.perf_counter()
    res = minimize(_obj, cfg['theta0'], method='COBYLA',
                   options={'maxiter': 1000, 'rhobeg': 0.3, 'catol': 1e-5})
    dt  = time.perf_counter() - t0
    optimized_thetas[ny] = res.x.tolist()
    cobyla_phi[ny] = res.fun
    err = abs(res.fun - classical_phi[ny]) / classical_phi[ny] * 100
    print(f'  φ_DQA={res.fun:.4f}  φ_class={classical_phi[ny]:.4f}  '
          f'err={err:.2f}%  evals={calls[0]}  t={dt:.1f}s')

# ── FINAL EVALUATION: ideal vs noisy ────────────────────────────────────────
print('\n── Final evaluation: ideal vs noisy ──')
results_ideal, results_noisy = {}, {}
for ny, thetas in optimized_thetas.items():
    cfg = CONFIGS[ny]
    print(f'\n  n_y={ny}...', flush=True)

    opt_i = CudaqQAEOptimizer(
        c_x=[3.]*ny, c_y=cfg['c_y'], c_r=cfg['c_r'],
        n_y=ny, w_d=cfg['w_d'], cost_norm=cfg['cost_norm'], noise_model=None)
    t0    = time.perf_counter()
    phi_i = opt_i.estimate_expected_value_observe(thetas)
    dt_i  = time.perf_counter() - t0
    results_ideal[ny] = phi_i

    opt_n = CudaqQAEOptimizer(
        c_x=[3.]*ny, c_y=cfg['c_y'], c_r=cfg['c_r'],
        n_y=ny, w_d=cfg['w_d'], cost_norm=cfg['cost_norm'], noise_model=noise_model)
    t0    = time.perf_counter()
    phi_n = opt_n.estimate_expected_value_noisy_trajectory(thetas, N_TRAJ)
    dt_n  = time.perf_counter() - t0
    results_noisy[ny] = phi_n

    ref = classical_phi[ny]
    deg = abs(phi_n - phi_i) / abs(phi_i) * 100
    print(f'  Ideal : φ={phi_i:.4f}  vs classical err={abs(phi_i-ref)/ref*100:.2f}%  ({dt_i*1e3:.0f} ms)')
    print(f'  Noisy : φ={phi_n:.4f}  vs classical err={abs(phi_n-ref)/ref*100:.2f}%  ({dt_n*1e3:.0f} ms)')
    print(f'  Noise degradation: {deg:.2f}%')

# ── PLOT ──────────────────────────────────────────────────────────────────────
ny_vals = N_Y_LIST
ref_v   = [classical_phi[ny]    for ny in ny_vals]
phi_i   = [results_ideal[ny]    for ny in ny_vals]
phi_n   = [results_noisy[ny]    for ny in ny_vals]
err_i   = [abs(results_ideal[ny]-classical_phi[ny])/classical_phi[ny]*100 for ny in ny_vals]
err_n   = [abs(results_noisy[ny]-classical_phi[ny])/classical_phi[ny]*100 for ny in ny_vals]
deg     = [abs(results_noisy[ny]-results_ideal[ny])/abs(results_ideal[ny])*100 for ny in ny_vals]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Left: absolute phi values
ax = axes[0]
ax.plot(ny_vals, ref_v, 'k--o', lw=2, ms=8, label='Classical optimal')
ax.plot(ny_vals, phi_i, 'b-s',  lw=2, ms=8, label='DQA ideal (p=2, noiseless)')
ax.plot(ny_vals, phi_n, 'r-^',  lw=2, ms=8, label=f'DQA noisy (p₁={P1}, p₂={P2})')
ax.set_xlabel('$n_y$ (turbines)', fontsize=12)
ax.set_ylabel('Expected second-stage cost  $\\phi$', fontsize=12)
ax.set_title('Expected cost vs system size', fontsize=13)
ax.set_xticks(ny_vals); ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

# Middle: error vs classical
ax = axes[1]
x, bw = np.arange(len(ny_vals)), 0.32
b1 = ax.bar(x-bw/2, err_i, width=bw, color='steelblue', alpha=0.85, label='Ideal')
b2 = ax.bar(x+bw/2, err_n, width=bw, color='tomato',    alpha=0.85, label=f'Noisy (p₁={P1}, p₂={P2})')
for rect, val in zip(list(b1)+list(b2), err_i+err_n):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
ax.set_xlabel('$n_y$', fontsize=12); ax.set_ylabel('Error vs classical  (%)', fontsize=11)
ax.set_title('DQA accuracy: ideal vs noisy', fontsize=13)
ax.set_xticks(x); ax.set_xticklabels([f'$n_y={n}$' for n in ny_vals])
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)

# Right: pure noise degradation
ax = axes[2]
bars = ax.bar(ny_vals, deg, color='darkorange', alpha=0.85, width=1.2)
for rect, val in zip(bars, deg):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.05,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('$n_y$ (turbines)', fontsize=12)
ax.set_ylabel('Noise degradation  (%)\n$= |\\phi_{noisy}-\\phi_{ideal}|/|\\phi_{ideal}|$', fontsize=10)
ax.set_title(f'Pure noise effect (p₁={P1}, p₂={P2})\n{N_TRAJ} trajectories', fontsize=13)
ax.set_xticks(ny_vals); ax.grid(True, axis='y', alpha=0.4)

plt.tight_layout()
out_png = os.path.join(_QISKIT_IMPL, 'noise_study_accuracy.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f'\nPlot saved → {out_png}')

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print(f"\n{'n_y':>4} {'φ_class':>10} {'φ_ideal':>9} {'err_i%':>8} "
      f"{'φ_noisy':>9} {'err_n%':>8} {'noise_deg%':>11}")
print('─'*65)
for ny in ny_vals:
    ref = classical_phi[ny]
    pi  = results_ideal[ny]
    pn  = results_noisy[ny]
    print(f"{ny:>4} {ref:>10.4f} {pi:>9.4f} {abs(pi-ref)/ref*100:>8.2f} "
          f"{pn:>9.4f} {abs(pn-ref)/ref*100:>8.2f} {abs(pn-pi)/abs(pi)*100:>11.2f}")

# Save JSON
out_json = os.path.join(_QISKIT_IMPL, 'noise_study_results.json')
json.dump({str(ny): {
    'classical': classical_phi[ny], 'ideal': results_ideal[ny],
    'noisy': results_noisy[ny], 'thetas': optimized_thetas[ny],
    'p_layers': P_LAYERS, 'p1': P1, 'p2': P2, 'n_traj': N_TRAJ,
} for ny in ny_vals}, open(out_json,'w'), indent=2)
print(f'Results saved → {out_json}')
