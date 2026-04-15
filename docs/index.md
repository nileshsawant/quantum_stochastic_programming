# Quantum Stochastic Programming

Quantum computing algorithms for **two-stage stochastic optimization**, with a focus on the Unit Commitment (UC) problem in power systems.
The algorithms combine **Discrete Quantum Annealing (DQA)** with **Quantum Amplitude Estimation (QAE)** to compute expected-value objective functions over a probability distribution of wind-power scenarios.

Based on: [arXiv 2402.15029](https://arxiv.org/abs/2402.15029) — *"Quantum algorithms for the two-stage stochastic unit commitment problem"*

---

## Problem Description

The **Unit Commitment** problem decides which gas generators to commit before uncertain wind power is realised (first-stage decision $x$), then dispatches all generators optimally for each realised scenario (second-stage recourse $y$).

$$\min_{x,y} \; c_x^\top x + \mathbb{E}_\xi\!\left[\min_r c_r r(\xi) \right] \quad \text{s.t.} \quad \sum_j y_j(\xi) + r(\xi) = d, \quad y_j(\xi) \le \xi_j$$

The quantum algorithm evaluates $\mathbb{E}[C(x,\xi)]$ via QAE in $O(1/\epsilon)$ circuit evaluations rather than the $O(1/\epsilon^2)$ shots needed classically.

---

## Repository Structure

Two implementations exist:

| | `qiskit_impl/` | `QuantumExpectedValueFunctionProject/` |
|---|---|---|
| API style | Functional primitives + `args` dicts | Class-based OOP |
| Encoding | Binary only | Binary or Unary |
| Qiskit version | 2.x (current) | Originally 0.x/1.x (needs import fixes) |
| Status | Active development | Legacy / reference |

---

## Quick Start

```python
from qiskit_impl.binary_optimizer import BinaryNestedOptimizer
import numpy as np

pdf = {(0,0):0.25, (0,1):0.25, (1,0):0.25, (1,1):0.25}
bno = BinaryNestedOptimizer([0.4], [0.05, 0.10], 1.0, pdf, demand=1, is_uniform=True)

# Classical reference
ev_true = bno.brute_force_wind_demand_expectation_values()
print("Classical optimal:", ev_true)

# DQA estimate
qc = bno.adiabatic_evolution_circuit(wind_demand=1, time=100, time_steps=4, norm=10)
counts = bno.execute_optimizer(qc, num_meas=4096)
ev_dqa = bno.process_expectation_value_optimizer(wind_demand=1, counts=counts)
print("DQA estimate:", ev_dqa)
```

---

## Installation

### NREL HPC (GPU-enabled)

```bash
module load qiskit/aer-gpu
# Python 3.12, Qiskit 2.2.3, qiskit-aer-gpu 0.15.1 (CUDA 12.4, H100)
```

### Local

```bash
pip install qiskit>=2.0 qiskit-aer qiskit-algorithms matplotlib scipy numpy
```

---

## Navigation

- **[Algorithm Overview](algorithm.md)** — How DQA and QAE work step by step
- **[API Reference](api/binary_optimizer.md)** — Full class and function documentation
- **[GPU Porting Guide](gpu_porting.md)** — Three-tier strategy for H100 acceleration
- **[References](references.md)** — Papers and further reading
