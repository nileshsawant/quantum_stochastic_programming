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

## Porting Qiskit Circuits to CUDA-Q

When translating a Qiskit circuit or gate to CUDA-Q, silent correctness errors
are common. The circuit compiles and runs without errors but produces wrong
results — often dismissed as numerical noise or shot noise. Follow this
checklist to catch them early.

### Rule 1: validate every ported gate with a statevector test

Before using a ported gate in a larger circuit, write a unit test that applies
it in isolation and compares the resulting statevector (or unitary matrix)
against the Qiskit reference.

```python
import cudaq, math, numpy as np
from qiskit.circuit.library import SwapGate

# --- CUDA-Q kernel under test ---
@cudaq.kernel
def my_gate(beta: float, q0: cudaq.qubit, q1: cudaq.qubit):
    # ... your implementation ...
    pass

# Reference unitary from Qiskit
qiskit_unitary = np.array(SwapGate().power(0.5).to_matrix())

# CUDA-Q unitary
cudaq.set_target("qpp-cpu")
cudaq_unitary = np.array(cudaq.get_unitary(my_gate, 0.5))

# Compare (up to global phase)
assert np.allclose(np.abs(cudaq_unitary), np.abs(qiskit_unitary), atol=1e-6), \
    "Gate unitary mismatch — porting error!"
```

Do this for **every** non-trivial gate before integrating it into the full
ansatz. Small errors compound over many layers and produce results that look
like noise but are not.

### Rule 2: common porting error — missing phases in parametric gates

Gates defined by a product of Pauli exponentials (e.g. fSWAP, iSWAP, XY gate
families) often have an **odd-parity phase** term that Qiskit includes but
naive decompositions omit. The symptom is a systematic bias in expectation
values that grows with circuit depth, indistinguishable from gate noise in
single-shot checks.

Example: `SwapGate().power(beta)` in Qiskit. A naive CUDA-Q port using only
Rxx and Ryy terms is **missing** the required CX–Rz–CX phase correction:

```python
@cudaq.kernel
def fswap_power(beta: float, q0: cudaq.qubit, q1: cudaq.qubit):
    angle = beta * math.pi / 2.0

    # Rxx(angle)
    h(q0);  h(q1)
    cx(q0, q1);  rz(angle, q1);  cx(q0, q1)
    h(q0);  h(q1)

    # Ryy(angle)
    rx(math.pi / 2.0, q0);  rx(math.pi / 2.0, q1)
    cx(q0, q1);  rz(angle, q1);  cx(q0, q1)
    rx(-math.pi / 2.0, q0);  rx(-math.pi / 2.0, q1)

    # Odd-parity phase — REQUIRED to match Qiskit's SwapGate().power(beta)
    # Omitting these three lines is the most common porting mistake for this gate
    cx(q0, q1)
    rz(angle, q1)
    cx(q0, q1)
```

Always derive the full decomposition from the gate's unitary definition, not
from a partial circuit identity.

### Rule 3: bit ordering is reversed between Qiskit and CUDA-Q

Qiskit uses **little-endian** ordering: qubit 0 is the **least significant bit**
(rightmost) in bitstrings. CUDA-Q uses **big-endian** ordering: qubit 0 is the
**most significant bit** (leftmost).

| Framework | Bitstring for |0⟩⊗|1⟩ (q0=0, q1=1) | Position of q0 |
|-----------|---------------------------------------|----------------|
| Qiskit    | `"10"` | LSB (right) |
| CUDA-Q    | `"01"` | MSB (left)  |

When post-processing measurement bitstrings from CUDA-Q, index from the left:
```python
# CUDA-Q: bitstring[0] is qubit 0
q0_val = int(bitstring[0])
q1_val = int(bitstring[1])

# Qiskit: bitstring[-1] is qubit 0
q0_val = int(bitstring[-1])
q1_val = int(bitstring[-2])
```

Mismatched indexing causes scrambled register assignments that are hard to
detect without a controlled test (e.g. prepare a known basis state and assert
the correct bit positions).

### Porting checklist

- [ ] Unit-test each ported gate's unitary against the Qiskit reference
- [ ] Confirm odd-parity / phase terms are included in all parametric gates
- [ ] Verify bitstring index conventions (big- vs little-endian) in all
      post-processing code
- [ ] Prepare a known basis state (e.g. |01⟩) and assert the correct bitstring
      is returned before running the full ansatz
- [ ] Cross-check expectation values / sample distributions for a small
      system (2–4 qubits) against Qiskit before scaling up

---

## Noisy Circuit Simulation

### Choosing the right evaluation method

| Method | API | Speed | Noise support | Notes |
|--------|-----|-------|---------------|-------|
| Statevector loop | `cudaq.get_state()` + Python loop | Slow — iterates 2ⁿ states in Python | No | Avoid for n ≥ 8 |
| Pauli observe | `cudaq.observe()` with spin Hamiltonian | Fast — single GPU call | **No** | Preferred for noiseless evaluation |
| Shot sampling | `cudaq.sample()` + post-processing | Medium | **Yes** | Required for noisy evaluation |

### Do not use `cudaq.observe()` for constraint-penalized cost functions under noise

`cudaq.observe()` computes a raw Pauli expectation value. It knows nothing
about constraint penalties defined in post-processing. If your cost function
adds a penalty for bitstrings that violate constraints (e.g. Hamming-weight
constraints), those penalties are absent when using `observe()` — so
off-constraint bitstrings appear artificially cheap, and you may observe
`cost_noisy < cost_ideal`, which is physically wrong.

**Always use shot sampling + manual post-processing for noisy evaluation of
penalized cost functions:**

```python
# cudaq.sample() with noise_model set on the target
counts = cudaq.sample(kernel, *params, noise_model=noise_model, shots_count=N_SHOTS)

cost_noisy = 0.0
for bitstring, count in counts.items():
    prob = count / N_SHOTS
    # Apply the FULL cost function including all penalty terms
    cost_noisy += full_cost(bitstring) * prob
```

Ensure `full_cost()` exactly matches the cost function used for the noiseless
reference — any missing penalty term creates a systematic downward bias in
the noisy estimate.

### Noise model parameters

Near-term device realistic values:
- `p1 ≈ 1e-4` (0.01%) — single-qubit gate depolarizing rate
- `p2 ≈ 1e-3` (0.1%) — two-qubit gate depolarizing rate

Using `p2 = 0.01` (1%) will produce noise levels that dominate the signal
entirely for circuits with more than ~100 two-qubit gates.

### Noise study across system sizes: use absolute metrics and fixed circuit depth

When benchmarking noise impact across different system sizes:

1. **Fix circuit depth** (number of layers / timesteps) to the same value for
   all sizes. If depth scales with system size, small systems have too few
   layers for the ansatz to converge — lack of expressivity dominates over
   noise and masks the noise signal, making the comparison meaningless.

2. **Use absolute metrics**: report `Δcost = cost_noisy − cost_ideal` (absolute
   noise-induced increase). Avoid relative metrics such as
   `Δcost / cost_ideal` — the denominator grows with system size, making noise
   appear to *decrease* even when the absolute impact is increasing.

3. **Visualization**: keep all three panels of a noise comparison figure in the
   same absolute cost units so they tell a consistent story. Panels using
   different denominators can visually contradict each other and are misleading.

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
