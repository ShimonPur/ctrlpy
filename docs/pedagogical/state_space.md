# State-Space Canonical Forms & Controllability/Observability Tutor

Modern control theory formulates linear time-invariant (LTI) dynamics in terms of internal state variables. Understanding the structural properties of state-space models—specifically **Controllability**, **Observability**, and **Canonical Similarity Transformations**—is foundational for state-feedback pole placement, observer (estimator) design, and minimal realization analysis.

`ctrlpy` provides an analytical, step-by-step state-space educational tutor in `ctrlpy.symbolic.state_space.StateSpaceTutor` (and `ctrlpy.pedagogy.state_space`), designed for both control engineering and classroom pedagogy.

---

## 1. Theoretical Background

Consider an $n$-th order continuous-time (or discrete-time) LTI state-space system:

$$\begin{aligned}
\dot{x}(t) &= A x(t) + B u(t) \\
y(t) &= C x(t) + D u(t)
\end{aligned}$$

where $x(t) \in \mathbb{R}^n$, $u(t) \in \mathbb{R}^m$, and $y(t) \in \mathbb{R}^p$.

### Controllability (Kalman & PBH Criteria)

A system is **controllable** if any initial state $x(0)$ can be steered to any target state $x(t_1)$ in finite time $t_1 > 0$ by an unconstrained input $u(t)$.

1. **Kalman Controllability Matrix:**
   $$\mathcal{C} = \begin{bmatrix} B & AB & A^2 B & \dots & A^{n-1} B \end{bmatrix} \in \mathbb{R}^{n \times nm}$$
   $$\text{System is controllable} \iff \operatorname{rank}(\mathcal{C}) = n$$

2. **Popov-Belevitch-Hautus (PBH) Eigenvalue Controllability Test:**
   $$\text{Eigenvalue } \lambda_i \text{ is controllable} \iff \operatorname{rank}\begin{bmatrix} \lambda_i I - A & B \end{bmatrix} = n$$

### Observability (Kalman & PBH Criteria)

A system is **observable** if the initial state $x(0)$ can be uniquely determined from knowledge of the input $u(t)$ and output $y(t)$ over a finite interval $[0, t_1]$.

1. **Kalman Observability Matrix:**
   $$\mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix} \in \mathbb{R}^{np \times n}$$
   $$\text{System is observable} \iff \operatorname{rank}(\mathcal{O}) = n$$

2. **PBH Eigenvalue Observability Test:**
   $$\text{Eigenvalue } \lambda_i \text{ is observable} \iff \operatorname{rank}\begin{bmatrix} \lambda_i I - A \\ C \end{bmatrix} = n$$

### Kalman 4-Subspace Mode Decomposition

Every LTI system can be decomposed into four decoupled orthogonal subspaces based on modal controllability and observability:

| Subspace | Controllable? | Observable? | Transfer Function Presence | Physical Interpretation |
|:---|:---:|:---:|:---:|:---|
| $\Sigma_{co}$ | Yes | Yes | **Included** in $G(s)$ | Directly actuated and sensed; governs input-output response. |
| $\Sigma_{c\bar{o}}$ | Yes | No | **Cancelled** (Hidden from $y$) | Actuated by $u(t)$, but sensor blind; internal pole-zero cancellation. |
| $\Sigma_{\bar{c}o}$ | No | Yes | **Cancelled** (Hidden from $u$) | Sensed by $y(t)$, but actuator blind; autonomous disturbance/drift. |
| $\Sigma_{\bar{c}\bar{o}}$ | No | No | **Cancelled** (Completely isolated) | Neither actuated nor sensed; decoupled internal dynamic mode. |

---

## 2. Analytical Canonical Transformations

State vectors under a nonsingular similarity transformation $x = T z$ ($z = T^{-1} x$) transform state-space matrices according to:

$$A_{\text{can}} = T^{-1} A T, \quad B_{\text{can}} = T^{-1} B, \quad C_{\text{can}} = C T, \quad D_{\text{can}} = D$$

`StateSpaceTutor` derives three standard canonical representations:

### 1. Controllable Canonical Form (Phase-Variable Form)

For an $n$-th order SISO system with characteristic polynomial $p(s) = s^n + a_{n-1}s^{n-1} + \dots + a_1 s + a_0$ and strictly proper transfer function $G(s) = \frac{b_{n-1}s^{n-1} + \dots + b_0}{s^n + a_{n-1}s^{n-1} + \dots + a_0}$:

$$A_c = \begin{bmatrix} 0 & 1 & 0 & \dots & 0 \\ 0 & 0 & 1 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & 1 \\ -a_0 & -a_1 & -a_2 & \dots & -a_{n-1} \end{bmatrix}, \quad B_c = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix}, \quad C_c = \begin{bmatrix} b_0 & b_1 & \dots & b_{n-1} \end{bmatrix}$$

- **Transformation Matrix:** $T_c = \mathcal{C} \mathcal{C}_c^{-1} = \mathcal{C} \mathcal{W}$
- **Condition:** Exists if and only if $\operatorname{rank}(\mathcal{C}) = n$.

### 2. Observable Canonical Form

The dual of the controllable canonical form:

$$A_o = \begin{bmatrix} 0 & 0 & \dots & 0 & -a_0 \\ 1 & 0 & \dots & 0 & -a_1 \\ 0 & 1 & \dots & 0 & -a_2 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & \dots & 1 & -a_{n-1} \end{bmatrix}, \quad B_o = \begin{bmatrix} b_0 \\ b_1 \\ \vdots \\ b_{n-1} \end{bmatrix}, \quad C_o = \begin{bmatrix} 0 & 0 & \dots & 0 & 1 \end{bmatrix}$$

- **Transformation Matrix:** $T_o = \mathcal{O}^{-1} \mathcal{O}_o = \mathcal{O}^{-1} \mathcal{C}_c$
- **Condition:** Exists if and only if $\operatorname{rank}(\mathcal{O}) = n$.

### 3. Jordan / Diagonal Modal Form

Transforms the system into decoupled modal state equations using the modal eigenvector matrix $V$:

$$A_d = V^{-1} A V = \operatorname{diag}(\lambda_1, \lambda_2, \dots, \lambda_n), \quad B_d = V^{-1} B, \quad C_d = C V$$

- Modal state evolution: $\dot{z}_i(t) = \lambda_i z_i(t) + b_{d,i} u(t)$
- Output synthesis: $y(t) = \sum_{i=1}^n c_{d,i} z_i(t) + D u(t)$
- Mode $\lambda_i$ is controllable $\iff b_{d,i} \ne 0$
- Mode $\lambda_i$ is observable $\iff c_{d,i} \ne 0$

---

## 3. Basic Python Usage

### Analyzing a Controllable and Observable System

```python
import ctrlpy as cp
from ctrlpy.symbolic import StateSpaceTutor, state_space_tutor

# Define a 2nd order state-space model: p(s) = s^2 + 3s + 2 = (s+1)(s+2)
A = [[0, 1], [-2, -3]]
B = [[0], [1]]
C = [[1, 0]]
D = [[0]]

tutor = StateSpaceTutor(A, B, C, D)

print(f"Controllable: {tutor.is_controllable} (Rank: {tutor.controllability_rank}/2)")
print(f"Observable:   {tutor.is_observable}   (Rank: {tutor.observability_rank}/2)")
print(f"Eigenvalues:  {tutor.eigenvalues}")
print(tutor)
```

Output:
```text
=== Pedagogical State-Space Analysis ===
State Dimension: n=2, Inputs: m=1, Outputs: p=1
A =
Matrix([[0, 1], [-2, -3]])
B =
Matrix([[0], [1]])
C =
Matrix([[1, 0]])
D =
Matrix([[0]])

Controllability Matrix C (Rank 2/2, Controllable=True):
Matrix([[0, 1], [1, -3]])

Observability Matrix O (Rank 2/2, Observable=True):
Matrix([[1, 0], [0, 1]])

Characteristic Polynomial: s**2 + 3*s + 2 = 0
Eigenvalues: [-1, -2]
Uncontrollable Modes: []
Unobservable Modes: []
```

---

## 4. Canonical Form Conversions

You can extract individual canonical transformation results directly:

```python
# Extract Controllable Canonical Form (Phase-Variable Form)
ccf = tutor.controllable_canonical_form()
print("A_c =", ccf.A)
print("B_c =", ccf.B)
print("C_c =", ccf.C)
print("Transformation T_c (x = T_c z_c):", ccf.T)

# Extract Observable Canonical Form
ocf = tutor.observable_canonical_form()
print("A_o =", ocf.A)
print("B_o =", ocf.B)
print("C_o =", ocf.C)
print("Transformation T_o (x = T_o z_o):", ocf.T)

# Extract Jordan / Diagonal Modal Form
jcf = tutor.jordan_canonical_form()
print("A_d =", jcf.A)
print("Modal Matrix V (x = V z_d):", jcf.T)
```

---

## 5. Detecting Pole-Zero Cancellations & Unobservable Modes

When a state-space realization contains internal cancellations, `StateSpaceTutor` details exactly which modes are hidden from the transfer function:

```python
# System with an unobservable mode at s = -5
A = [[-2, 0], [0, -5]]
B = [[1], [1]]
C = [[1, 0]]
D = [[0]]

tutor = state_space_tutor(A, B, C, D)

print(f"Is Controllable: {tutor.is_controllable}")  # True
print(f"Is Observable:   {tutor.is_observable}")  # False (Rank 1/2)
print(f"Unobservable Modes: {tutor.unobservable_modes}")  # [-5]
print(f"Input-Output Transfer Function: G(s) = {tutor.transfer_function}")  # 1/(s + 2)

# Inspect PBH mode breakdown
for mode in tutor.modes:
    print(f"Mode λ = {mode.eigenvalue}: {mode.description}")
```

---

## 6. Step-by-Step Derivation Trace (`.explain_steps()`)

In educational environments and Jupyter notebooks, `.explain_steps()` generates formatted derivations of all 8 phases:

```python
steps = tutor.explain_steps()
for step in steps:
    print(step)
    print("-" * 60)
```

---

## 7. API Quick Reference

| Class / Function | Description |
|:---|:---|
| [`StateSpaceTutor`](../api/pedagogy.md) | Primary tutor engine for analytical state-space derivation, PBH tests, and canonical transformations. |
| [`CanonicalFormResult`](../api/pedagogy.md) | Structured container holding canonical matrices $(A_{\text{can}}, B_{\text{can}}, C_{\text{can}}, D)$ and transformation matrices $T, T^{-1}$. |
| [`ModeAnalysis`](../api/pedagogy.md) | Dataclass summarizing PBH controllability/observability rank and Kalman classification for a single pole. |
| `state_space_tutor(A, B, C, D)` | Convenience constructor for `StateSpaceTutor`. |
| `controllable_canonical_form(sys)` | Direct helper returning the Controllable Canonical Form. |
| `observable_canonical_form(sys)` | Direct helper returning the Observable Canonical Form. |
| `jordan_canonical_form(sys)` | Direct helper returning the Jordan / Diagonal Modal Form. |
| `controllability_matrix(sys)` | Direct helper returning the Kalman controllability matrix $\mathcal{C}$. |
| `observability_matrix(sys)` | Direct helper returning the Kalman observability matrix $\mathcal{O}$. |
