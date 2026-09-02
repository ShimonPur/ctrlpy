# Routh-Hurwitz Stability Criterion

The **Routh-Hurwitz Stability Criterion** is a classical algebraic technique for determining the stability of continuous-time Linear Time-Invariant (LTI) systems without explicitly computing the roots of their characteristic polynomial.

`ctrlpy` provides an analytical, step-by-step implementation in `ctrlpy.symbolic.routh` (`RouthArray`), designed for both practical engineering and classroom pedagogy.

---

## 1. Theoretical Background

Consider an $n$-th order continuous-time characteristic polynomial:

$$P(s) = a_n s^n + a_{n-1} s^{n-1} + a_{n-2} s^{n-2} + \dots + a_1 s + a_0 = 0$$

### Array Construction

The Routh table is formed by arranging the coefficients into rows:

$$\begin{array}{c|cccc}
s^n & a_n & a_{n-2} & a_{n-4} & \dots \\
s^{n-1} & a_{n-1} & a_{n-3} & a_{n-5} & \dots \\
s^{n-2} & b_1 & b_2 & b_3 & \dots \\
s^{n-3} & c_1 & c_2 & c_3 & \dots \\
\vdots & \vdots & \vdots & \vdots & \ddots \\
s^0 & a_0 & 0 & 0 & \dots
\end{array}$$

where intermediate entries are computed via cross-multiplication:

$$b_1 = \frac{a_{n-1} a_{n-2} - a_n a_{n-3}}{a_{n-1}}, \quad b_2 = \frac{a_{n-1} a_{n-4} - a_n a_{n-5}}{a_{n-1}}$$

$$c_1 = \frac{b_1 a_{n-3} - a_{n-1} b_2}{b_1}, \quad c_2 = \frac{b_1 a_{n-5} - a_{n-1} b_3}{b_1}$$

### The Routh-Hurwitz Theorem

- **Asymptotic Stability**: The polynomial $P(s)$ is strictly Hurwitz (all roots in the open Left-Half Plane, $\operatorname{Re}(s) < 0$) if and only if **all elements in the first column of the Routh array have the same sign** (and none are zero).
- **RHP Poles**: The number of roots with positive real parts ($\operatorname{Re}(s) > 0$) is equal to the **number of sign changes** in the first column.

---

## 2. Handling Textbook Special Cases

`RouthArray` handles standard edge cases automatically:

### Special Case 1: First Element in a Row is Zero

When the first entry in a row $s^k$ is zero but the remaining row contains non-zero entries, direct division by zero is avoided by replacing the zero with a small symbolic parameter $\epsilon > 0$:

$$0 \longrightarrow \epsilon > 0$$

The remaining rows are derived in terms of $\epsilon$, and the signs of column 1 entries are evaluated by computing the right-sided limit $\lim_{\epsilon \to 0^+} \operatorname{sgn}(\text{entry})$.

### Special Case 2: Row of All Zeros

A row of all zeros indicates the presence of roots symmetric about the origin in the complex $s$-plane (such as pure imaginary poles $\pm j\omega_0$, real symmetric poles $\pm \sigma_0$, or complex quadruplets $\pm \sigma \pm j\omega$).

When row $s^k$ is all zeros:
1. Form an **Auxiliary Polynomial** $A(s)$ from the preceding row $s^{k+1}$.
2. Compute the formal derivative $\frac{dA(s)}{ds}$.
3. Replace the zero row $s^k$ with the coefficients of $\frac{dA(s)}{ds}$ and continue array construction.

---

## 3. Basic Python Usage

Import `RouthArray` or convenience function `routh_table` from `ctrlpy.symbolic`:

```python
import ctrlpy as cp
from ctrlpy.symbolic import RouthArray, routh_table

# Analyze characteristic polynomial P(s) = s^3 + 2s^2 + 3s + 4
ra = RouthArray([1, 2, 3, 4])

print(f"Is strictly stable: {ra.is_stable}")
print(f"Number of RHP poles: {ra.num_rhp_poles}")
print(ra)
```

Output:
```text
=== Routh-Hurwitz Stability Criterion ===
Polynomial: s**3 + 2*s**2 + 3*s + 4 = 0

s^3 | 1  3
s^2 | 2  4
s^1 | 1  0
s^0 | 4  0
------------------------------------------
Number of RHP Poles: 0
Asymptotically Stable: True
```

---

## 4. Parametric Gain Stability Bounds ($K$)

You can evaluate the stability range of a controller gain $K$:

```python
import sympy as sp
from ctrlpy.symbolic import RouthArray

s, K = sp.symbols("s K")

# Closed-loop characteristic equation: s^3 + 3s^2 + 3s + (1 + K) = 0
char_eq = s**3 + 3 * s**2 + 3 * s + 1 + K

ra = RouthArray(char_eq, k_symbol="K")
print(f"Stable Gain Range for K: {ra.k_range}")
```

Output:
```text
Stable Gain Range for K: -1 < K < 8
```

You can also pass a `TransferFunction` directly:

```python
import ctrlpy as cp
from ctrlpy.symbolic import RouthArray

# Open-loop plant G(s) = 1 / (s(s+1)(s+2))
G = cp.tf([1], [1, 3, 2, 0])

# Closed-loop with unity feedback and proportional gain K
ra = RouthArray(G, k_symbol="K")
print(f"Stable K Range: {ra.k_range}")  # 0 < K < 6
```

---

## 5. Pedagogical Derivation Steps (`.explain_steps()`)

To view a detailed mathematical breakdown suitable for lecture slides or student feedback, call `.explain_steps()`:

```python
ra = RouthArray([1, 2, 3, 4])

for step in ra.explain_steps():
    print(f"- {step}")
```

Output:
```text
- **Characteristic Equation:** $P(s) = s^{3} + 2 s^{2} + 3 s + 4 = 0$ (Degree $n = 3$)
- **Row $s^3$ Initialization:** [$1$, $3$] (alternating even/odd coefficients)
- **Row $s^2$ Initialization:** [$2$, $4$] (alternating coefficients)
- **Row $s^1$ Computation:** [$1$, $0$]
- **Row $s^0$ Computation:** [$4$, $0$]
- **First Column Entries:** [$1$, $2$, $1$, $4$] with signs [1, 1, 1, 1] $\implies$ **0 sign change(s)**.
- **Stability Conclusion:** Strictly Asymptotically Stable (All roots in open LHP).
```

---

## 6. Jupyter Notebook Rich LaTeX Rendering

When evaluated in a Jupyter notebook cell, `RouthArray` automatically renders using MathJax / LaTeX:

```python
ra = RouthArray([1, 1, 2, 24])
ra  # Automatically calls _repr_latex_()
```

Renders as:

$$\begin{aligned}
\textbf{Routh-Hurwitz Array:} & \\
\begin{array}{c|cc}
s^3 & 1 & 2 \\
s^2 & 1 & 24 \\
s^1 & -22 & 0 \\
s^0 & 24 & 0
\end{array} & \\
\textbf{Stability Status:} &\quad \text{Unstable / Marginally Stable} \\
\textbf{RHP Poles:} &\quad 2
\end{aligned}$$

