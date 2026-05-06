---
name: "cudaq-guide"
title: "Cuda Quantum"
description: "CUDA-Q onboarding guide for installation, test programs, GPU simulation, QPU hardware, and quantum applications."
version: "1.0.0"
author: "Sachin Pisal <spisal@nvidia.com>"
tags: [cuda-quantum, quantum-computing, onboarding, getting-started, nvidia]
tools: [Read, Glob, Grep, Bash]
license: "Apache License 2.0"
compatibility: "Python 3.10+, C++ 20"
metadata:
    author: "Sachin Pisal <spisal@nvidia.com>"
    tags:
        - cuda-quantum
        - quantum-computing
        - onboarding
        - getting-started
        - nvidia
    languages:
        - python
        - c++
    domain: "quantum"
---

## CUDA-Q Getting Started Guide

You are a CUDA-Q expert assistant. Guide the user through the CUDA-Q platform
based on their `$ARGUMENTS`. If no argument is given, present the full
onboarding menu.

## Purpose

Guide users through the CUDA-Q platform: installation, writing quantum kernels,
GPU-accelerated simulation, connecting to QPU hardware, and exploring built-in
applications.

## Prerequisites

- Python 3.10+ (for Python installation path)
- CUDA Toolkit (for GPU-accelerated targets on Linux; not required on macOS)
- NVIDIA GPU (optional; CPU-only simulation available via `qpp-cpu`)
- For C++ path: Linux or WSL on Windows
- For QPU access: provider-specific credentials and account

## Instructions

- Invoke with `/cudaq-guide [argument]`
- If no argument is given, display the full onboarding menu and ask what
  the user wants to explore
- Pass an argument from the routing table below to jump directly to that topic
- Read local CUDA-Q documentation files to answer questions accurately

## References

| Section | Doc file |
| --- | --- |
| Install | `docs/sphinx/using/install/install.rst`, `docs/sphinx/using/quick_start.rst` |
| Test Program | `docs/sphinx/using/basics/kernel_intro.rst`, `docs/sphinx/using/basics/build_kernel.rst` |
| GPU Simulation | `docs/sphinx/using/backends/sims/svsims.rst`, `docs/sphinx/using/examples/multi_gpu_workflows.rst` |
| QPU | `docs/sphinx/using/backends/hardware.rst`, `docs/sphinx/using/backends/cloud.rst` |
| Applications | `docs/sphinx/using/applications.rst` |
| Parallelize | `docs/sphinx/using/examples/multi_gpu_workflows.rst` |

## Routing by Argument

| Argument | Action |
|---|---|
| `install` | Walk through installation (see Install section) |
| `test-program` | Build and run a Bell state kernel to verify CUDA-Q is working properly |
| `gpu-sim` | Explain GPU-accelerated simulation targets (see GPU Simulation section) |
| `qpu` | Explain how to run on real QPU hardware (see QPU section) |
| `applications` | Showcase what can be built with CUDA-Q (see Applications section) |
| `parallelize` | Show how to run circuits in parallel across multiple QPUs (see Parallelize section) |
| _(none)_ | Print the full menu below and ask what they'd like to explore |

---

## Full Menu (no argument)

Present this when invoked with no argument

```text
CUDA-Q Getting Started

CUDA-Q is NVIDIA's unified quantum-classical programming model for CPUs, GPUs, and QPUs.
Supports Python and C++. Docs https://nvidia.github.io/cuda-quantum/

Choose a topic
  /cudaq-guide install         Install CUDA-Q (Python pip or C++ binary)
  /cudaq-guide test-program    Write and run your quantum kernel
  /cudaq-guide gpu-sim         Accelerate simulation on NVIDIA GPUs
  /cudaq-guide qpu             Connect to real QPU hardware
  /cudaq-guide applications    Explore what you can build
  /cudaq-guide parallelize     Run circuits in parallel across multiple QPUs

Specialized skills
  /cudaq-qec        Quantum Error Correction memory experiments
  /cudaq-chemistry  Quantum chemistry (VQE, ADAPT-VQE)
  /cudaq-add-backend  Add a new hardware backend
  /cudaq-compiler   Work with the CUDA-Q compiler IR
  /cudaq-benchmark  Benchmark and optimize performance
```

---

## Install

Instructions

- Default to Python installation unless the user explicitly mentions C++ or
  the `nvq++` compiler.
- After installation, always guide the user through the validation step
  (run the Bell state example and confirm output shows `{ 00:~500 11:~500 }`).
- Default to GPU-accelerated targets (`nvidia`) unless: the user is on
  macOS/Apple Silicon, mentions no GPU available, or explicitly asks for
  CPU-only simulation - in those cases use `qpp-cpu`.
- Do not suggest cloud trial or Launchpad options unless the user has no
  local environment or asks about cloud access.

Platform notes

- Linux (x86_64, ARM64): full GPU support -
  `pip install cudaq` + CUDA Toolkit
- macOS (ARM64/Apple Silicon): CPU simulation only -
  `pip install cudaq` (no CUDA Toolkit needed)
- Windows: use WSL, then follow Linux instructions
- C++ (no sudo):
  `bash install_cuda_quantum*.$(uname -m) --accept -- --installpath $HOME/.cudaq`
- Brev (cloud, no local setup): Log in at the NVIDIA Application Hub,
  open a CUDA-Q workspace, then SSH in with the Brev CLI:

  ```bash
  brev open ${WORKSPACE_NAME}
  ```

  CUDA-Q and the CUDA Toolkit are pre-installed.

---

## Test Program

Key concepts to explain

- `@cudaq.kernel` / `__qpu__` marks a quantum kernel - compiled to Quake MLIR
- `cudaq.qvector(N)` allocates N qubits in |0⟩
- `cudaq.sample()` - kernel measures qubits; returns bitstring histogram
  (`SampleResult`)
- `cudaq.run()` - kernel returns a classical value; runs `shots_count` times
  and returns a list of those return values
- `cudaq.observe()` - computes expectation value ⟨H⟩ for a spin operator
- `cudaq.get_state()` - returns the full statevector (simulator only)

Kernel restrictions

- Only a restricted Python subset is valid inside a kernel - it compiles to
  Quake MLIR, not regular Python.
- NumPy and SciPy cannot be used inside a kernel. Use them outside the kernel
  for classical pre/post-processing.
- Kernels can call other kernels; the callee must also be a `@cudaq.kernel`.

For compiler internals (`inspect` module -> `ast_bridge.py` -> Quake MLIR ->
QIR -> JIT), route to `/cudaq-compiler`.

---

## GPU Simulation

To recommend the best simulation backend for the user, consult the full
comparison table at
<https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html>

### Available GPU Targets

| Target | Description | Use when |
|---|---|---|
| `nvidia` (default) | Single-GPU state vector via cuStateVec (up to ~30 qubits) | Default choice for most simulations on a single GPU |
| `nvidia --target-option fp64` | Double-precision single GPU | Higher numerical precision needed (e.g. chemistry, sensitive observables) |
| `nvidia --target-option mgpu` | Multi-GPU, pools memory across GPUs (>30 qubits) | Circuit exceeds single-GPU memory; requires MPI |
| `nvidia --target-option mqpu` | Multi-QPU, one virtual QPU per GPU, parallel execution | Running many independent circuits in parallel (e.g. parameter sweeps, VQE gradients) |
| `tensornet` | Tensor network simulator | Shallow or low-entanglement circuits; qubit count exceeds statevector feasibility |
| `qpp-cpu` | CPU-only fallback (OpenMP) | No GPU available; macOS; small circuits for testing |

---

## QPU

When the user invokes this section, do not dump all providers at once.
Instead, follow this two-step dialogue:

Step 1 - ask which technology they want

```text
Which QPU technology are you targeting?
  1. Ion trap       (IonQ, Quantinuum)
  2. Superconducting (IQM, OQC, Anyon, TII, QCI)
  3. Neutral atom   (QuEra, Infleqtion, Pasqal)
  4. Cloud / multi-platform (AWS Braket, Scaleway)
```

Step 2 - once they pick a technology, ask which provider, then read the
corresponding doc file and walk the user through it step by step.

| Technology | Provider | Doc file |
|---|---|---|
| Ion trap | IonQ | `docs/sphinx/using/backends/hardware/iontrap.rst` (IonQ section) |
| Ion trap | Quantinuum | `docs/sphinx/using/backends/hardware/iontrap.rst` (Quantinuum section) |
| Superconducting | IQM | `docs/sphinx/using/backends/hardware/superconducting.rst` (IQM section) |
| Superconducting | OQC | `docs/sphinx/using/backends/hardware/superconducting.rst` (OQC section) |
| Superconducting | Anyon | `docs/sphinx/using/backends/hardware/superconducting.rst` (Anyon section) |
| Superconducting | TII | `docs/sphinx/using/backends/hardware/superconducting.rst` (TII section) |
| Superconducting | QCI | `docs/sphinx/using/backends/hardware/superconducting.rst` (QCI section) |
| Neutral atom | Infleqtion | `docs/sphinx/using/backends/hardware/neutralatom.rst` (Infleqtion section) |
| Neutral atom | QuEra | `docs/sphinx/using/backends/hardware/neutralatom.rst` (QuEra section) |
| Neutral atom | Pasqal | `docs/sphinx/using/backends/hardware/neutralatom.rst` (Pasqal section) |
| Cloud | AWS Braket | `docs/sphinx/using/backends/cloud/braket.rst` |
| Cloud | Scaleway | `docs/sphinx/using/backends/cloud/scaleway.rst` |

After walking through the provider steps, always close with

- Test locally first with `emulate=True` before submitting to real hardware.
- Use `cudaq.sample_async()` / `cudaq.observe_async()` for non-blocking submission.

---

## Applications

CUDA-Q ships with ready-to-run application notebooks

| Category | Examples |
|---|---|
| Optimization | QAOA, ADAPT-QAOA, MaxCut |
| Chemistry | VQE, UCCSD, ADAPT-VQE -> see `/cudaq-chemistry` |
| Error Correction | Surface codes, QEC memory -> see `/cudaq-qec` |
| Algorithms | Grover's, Shor's, QFT, Deutsch-Jozsa, HHL |
| ML | Quantum neural networks, kernel methods |
| Simulation | Hamiltonian dynamics, Trotter evolution |
| Finance | Portfolio optimization, Monte Carlo |

Point to sub-skills for specialized topics

- `/cudaq-qec` - full QEC memory experiment walkthrough
- `/cudaq-chemistry` - VQE and ADAPT-VQE for molecular energies
- `/cudaq-benchmark` - performance profiling and multi-GPU scaling

---

## Parallelize

CUDA-Q supports two distinct multi-GPU parallelization strategies - pick based
on what you are trying to scale.

| Goal | Strategy | Target option |
|---|---|---|
| Single circuit too large for one GPU | Pool GPU memory | `nvidia --target-option mgpu` |
| Many independent circuits at once | Run circuits in parallel | `nvidia --target-option mqpu` |
| Large Hamiltonian expectation value | Distribute terms across GPUs | `mqpu` + `execution=cudaq.parallel.thread` |

### Circuit batching with mqpu (`sample_async` / `observe_async`)

The `mqpu` option maps one virtual QPU to each GPU. Dispatch circuits
asynchronously with `qpu_id` to all GPUs simultaneously.

```python
import cudaq

cudaq.set_target("nvidia", option="mqpu")
n_qpus = cudaq.get_platform().num_qpus()

futures = [
    cudaq.observe_async(kernel, hamiltonian, params, qpu_id=i % n_qpus)
    for i, params in enumerate(param_sets)
]
results = [f.get().expectation() for f in futures]
```

### Hamiltonian batching

For a single kernel with a large Hamiltonian, add `execution=` to
`cudaq.observe` — no other code change needed.

```python
# Single node, multiple GPUs
result = cudaq.observe(kernel, hamiltonian, *args,
                       execution=cudaq.parallel.thread)

# Multi-node via MPI
result = cudaq.observe(kernel, hamiltonian, *args,
                       execution=cudaq.parallel.mpi)
```

See the docs above for complete working examples of both patterns.

---

## Noisy Circuit Simulation and Noise Studies

This section captures practical lessons from running depolarizing noise studies
on DQA (Digitized Quantum Annealing) circuits for stochastic programming
problems using CUDA-Q on an H100 GPU (`cuStateVec`).

### Building a depolarizing noise model

```python
from cudaq_impl import build_depolarizing_noise_model

noise_model = build_depolarizing_noise_model(p1=0.0001, p2=0.001)
# p1: single-qubit gate error probability (0.01%)
# p2: two-qubit gate error probability (0.1%)
# Realistic near-term device parameters
```

Pass `noise_model` to `CudaqQAEOptimizer`:
```python
optimizer = CudaqQAEOptimizer(..., noise_model=noise_model)
```

### Choosing the right evaluation method

CUDA-Q offers several ways to compute expected values — choose carefully:

| Method | API | Speed | Noise support | Notes |
|--------|-----|-------|---------------|-------|
| Statevector loop | `cudaq.get_state()` + Python loop | **Slow** for n_y ≥ 8 (iterates 2^n states in Python) | No | Avoid for large systems |
| Pauli observe | `cudaq.observe()` with spin Hamiltonian | **Fast** (single GPU call) | No (noiseless only) | Preferred for ideal evaluation |
| Shot sampling | `cudaq.sample()` + post-processing | Medium | **Yes** | Required for noisy evaluation; apply constraint penalties manually |

**Ideal evaluation** — use `cudaq.observe()` (single GPU kernel call, exact for noiseless):
```python
# estimate_expected_value_observe(thetas) in the project
result = cudaq.observe(kernel, hamiltonian, *thetas)
expectation = result.expectation()
```

**Noisy evaluation** — use `sample_ansatz(noise_model=..., shots=N_SHOTS)` with
manual post-processing. **Do not use `cudaq.observe()` with a noise model for
constraint-penalized cost functions** — `cudaq.observe()` computes raw Pauli
expectation values and has no concept of constraint penalties, so off-constraint
bitstrings appear artificially cheap and `φ_noisy < φ_ideal` for deep circuits.

Correct noisy evaluation pattern:
```python
counts = sample_ansatz(thetas, noise_model=noise_model, shots=N_SHOTS)
phi_noisy = 0.0
for bitstring, count in counts.items():
    prob = count / N_SHOTS
    y_bits  = bitstring[:n_y]   # turbine decisions
    xi_bits = bitstring[n_y:]   # wind scenarios
    true_output = y_bits * xi_bits
    op_cost  = float(np.dot(c_y, true_output))
    shortfall = max(0, w_d - int(true_output.sum()))
    phi_noisy += (op_cost + shortfall * c_r) * prob
```

### Known bug: `_wind_scenario_cost` missing shortfall penalty

`CudaqQAEOptimizer._wind_scenario_cost()` only sums operational costs and
**does not include the shortfall penalty** `max(0, w_d - Σ y_j·ξ_j) · c_r`.
This causes off-constraint bitstrings to appear cheap under noise, producing
`φ_noisy < φ_classical` — a physically wrong result.

Always apply the shortfall penalty manually in post-processing when running
a noise study (as shown above). The underlying `cudaq_impl.py` bug is tracked
separately.

### DQA linear-ramp ansatz

The digitized quantum annealing (DQA) linear-ramp parameter schedule:

```python
TIMESTEPS = 20   # fix for ALL system sizes (see fair comparison note below)
thetas = []
for t in range(TIMESTEPS):
    gamma_t = t / TIMESTEPS
    beta_t  = (1 - t / TIMESTEPS) / math.pi
    thetas += [gamma_t, beta_t]
# Total: 2*TIMESTEPS parameters (40 for TIMESTEPS=20)
```

### Fair comparison across system sizes

**Always fix `TIMESTEPS` to a common value** when comparing noise impact across
system sizes. If `timesteps = n_y`, small systems have too few DQA steps and
lack of expressivity dominates over noise — masking the noise signal entirely.

With `TIMESTEPS = 20` fixed: ideal DQA errors drop to **< 3%** for all
`n_y ∈ {4, 6, 8, 10}`, and the noise-induced cost increase becomes physically
meaningful and monotonically growing.

### Interpreting noise study results: use absolute metrics

The absolute noise-induced cost increase `Δφ = φ_noisy − φ_ideal` is the
correct metric for comparing noise impact across system sizes. **Do not use
relative metrics** such as `(φ_noisy − φ_ideal)/φ_ideal` — the denominator
grows with system size faster than `Δφ`, making noise appear to *decrease*
with `n_y` even when the absolute impact is increasing.

| Metric | Trend with n_y | Interpretation |
|--------|---------------|----------------|
| `Δφ = φ_noisy − φ_ideal` | ↑ monotonically | **Correct** — more noise damage for larger circuits |
| `Δφ / φ_ideal × 100%` | ↓ (misleading) | Denominator grows faster than numerator |

The cause is circuit-depth scaling: with fixed TIMESTEPS=20, 2-qubit gate
count scales as ~`20 · n_y²/2`, so the probability of at least one gate error
grows with `n_y`:

| n_y | ~2Q gates | P(≥1 error) at p₂=0.001 |
|-----|-----------|--------------------------|
| 4   | ~160      | ~15%                     |
| 6   | ~360      | ~30%                     |
| 8   | ~640      | ~47%                     |
| 10  | ~1000     | ~63%                     |

### Visualization: three-panel noise study plot

A three-panel figure communicates noise study results clearly:

1. **Left** — Line plot of absolute `φ` values (classical, ideal DQA, noisy DQA)
   vs `n_y`. The growing gap between ideal and noisy lines is the primary result.
2. **Middle** — Bar chart of `Δφ = φ_noisy − φ_ideal` per system size.
   Must increase monotonically — if it does not, check for evaluation bugs
   (missing shortfall penalty, wrong evaluator).
3. **Right** — Stacked bar decomposing `φ_noisy` into: classical optimal +
   DQA approximation gap + noise impact. The noise-impact segment (red) should
   visually grow with `n_y`, consistent with the left panel.

Avoid middle/right panels that use ratios with `φ_ideal` or `φ_classical` as
denominators — they produce visually decreasing bars that contradict the left
panel's message.

---

## Limitations

- GPU simulation requires Linux (x86_64 or ARM64); macOS is CPU-only
- Multi-GPU `mgpu` target requires MPI
- Kernel code must use a restricted Python subset; NumPy/SciPy are not
  allowed inside kernels
- QPU access requires provider-specific credentials and accounts

## Troubleshooting

- Import error after `pip install cudaq`: Ensure Python 3.10+ and a
  supported OS (Linux or macOS)
- No GPU detected: Verify CUDA Toolkit is installed and `nvidia-smi`
  shows your GPU; fall back to `qpp-cpu`
- Kernel compile error: Check that only supported Python constructs are
  used inside `@cudaq.kernel`
- QPU submission fails: Confirm credentials are set as environment
  variables per the provider docs
