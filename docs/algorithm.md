# Algorithm Overview

## Two-Stage Stochastic Optimization

The algorithm solves a two-stage stochastic program:

- **First stage** ($x$): Commit gas generators before wind is realized. Cost = $c_x^\top x$.
- **Second stage** ($y, r$): Given $x$ and realized wind $\xi$, choose turbine dispatch $y$ and recourse $r$ to meet demand $d$. Cost = $\sum_j c_y^j y_j \xi_j + c_r r$.
- **Objective**: $\min_x \left[ c_x^\top x + \mathbb{E}_\xi[\min_y q(x, y, \xi)] \right]$

The key difficulty is computing $\phi(x) = \mathbb{E}_\xi[\min_y q(x,y,\xi)]$: the min is *inside* the expectation, making classical Monte Carlo require $O(1/\epsilon^2)$ samples.

---

## Step 1: DQA — Discrete Quantum Annealing

DQA prepares a quantum state $|\tilde{\psi}_x^*\rangle$ that approximates the optimal second-stage decisions in superposition.

**Initialization**: Dicke state $|D_n^k\rangle$ — uniform superposition over all $\binom{n}{k}$ feasible turbine commitments:

$$|D_n^k\rangle = \frac{1}{\sqrt{\binom{n}{k}}} \sum_{|y|=k} |y\rangle \otimes |\xi\rangle_{\text{PDF}}$$

**Annealing**: Alternating layers of cost operator $U_q(\gamma)$ and XY mixer $U_d(\beta)$:

$$U_{\text{DQA}}(T) = \prod_{t=0}^{T-1} U_q(s_t) \cdot U_d(1 - s_t), \quad s_t = t/T$$

- $U_q(\gamma)$ encodes second-stage costs as phases via CP gates (Eq. 42)
- $U_d(\beta)$ uses SWAP$^\beta$ to mix turbine decisions while preserving $\sum y_j = k$ (Eq. 39)

**Output**: $|\tilde{\psi}_x^*\rangle$ with expected energy $\langle H_Q \rangle = \phi(x) + \delta$ where $\delta \geq 0$ is the residual temperature.

---

## Step 2: Oracle — Encoding cost as amplitude

The oracle $\mathcal{F}$ rotates an ancilla qubit so that:

$$\mathcal{F}|y\rangle|\xi\rangle|0\rangle_\text{anc} = |y\rangle|\xi\rangle\!\left(\sqrt{1-\bar{q}}\,|0\rangle + \sqrt{\bar{q}}\,|1\rangle\right)$$

where $\bar{q} = q(y,\xi)/q_{\max} \in [0,1]$. After $\mathcal{F}$: $\Pr[\text{ancilla}=|1\rangle] = \bar{q}$.

Two oracle variants:

| Oracle | Gates | Accuracy |
|---|---|---|
| $\mathcal{F}_\text{exact}$ | $O(2^{2n_y})$ multi-controlled RY | Exact |
| $\mathcal{F}_\text{sin}$ | $O(n_y)$ CCRY pairs | Approximate (small-angle) |

---

## Step 3: QAE — Quantum Amplitude Estimation

QAE estimates $a = \mathbb{E}[\bar{q}]$ (the normalized expected cost) with error $O(1/M)$ using $M$ circuit evaluations — quadratically better than classical Monte Carlo.

**Grover operator**: $\mathcal{Q} = \mathcal{A} S_0 \mathcal{A}^\dagger S_{\psi_0}$ rotates the state by $2\theta_a$ per application, where $a = \sin^2(\theta_a)$.

**QPE circuit** (Fig. 2 of paper):

1. $H^{\otimes m}$ on $m$ estimate qubits
2. Apply $\mathcal{A} = \mathcal{F} \cdot U_\text{DQA}$ once
3. Controlled-$\mathcal{Q}^{2^j}$ for $j = 0, \ldots, m-1$
4. Inverse QFT on estimate qubits
5. Measure $b$ → $\tilde{a} = \sin^2(b\pi/2^m)$ → $\tilde{\phi}(x) = \tilde{a} \cdot q_{\max}$

**Complexity**: $M = 2^m$ circuit evaluations, error $O(1/M)$ vs classical $O(1/\sqrt{M})$.

---

## Convergence

As $T \to \infty$ (more DQA steps), the residual $\delta \to 0$ and $\tilde{\phi}(x) \to \phi(x)$.
As $m \to \infty$ (more QPE qubits), QAE error $\to 0$ with only $M = 2^m$ total evaluations.

See [Fig. 4 in arXiv:2402.15029](https://arxiv.org/abs/2402.15029) for empirical convergence curves.
