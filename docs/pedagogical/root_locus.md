# Analytical Root Locus Rules

The **Evans Root Locus Method** is a classical graphical technique that plots the trajectories of closed-loop system poles in the complex $s$-plane as a scalar controller gain $K$ varies from $0$ to $+\infty$.

While numerical engines (such as `ctrlpy.plotting.plot_root_locus` and `ctrlpy.plotting_plotly.plot_root_locus_plotly`) solve for pole locations across thousands of gain values using eigenvalue routines, the pedagogical engine `ctrlpy.symbolic.root_locus.RootLocusRules` derives every classical analytical rule step-by-step with exact closed-form expressions.

---

## 1. Theoretical Foundation

Consider a standard unity feedback closed-loop system with open-loop transfer function:

$$L(s) = K G(s)H(s) = K \frac{N(s)}{D(s)}$$

The closed-loop characteristic equation is:

$$1 + K G(s)H(s) = 0 \iff G(s)H(s) = -\frac{1}{K}$$

This yields two fundamental conditions satisfied by all points $s$ on the root locus:

1. **Magnitude Condition:**
   $$|G(s)H(s)| = \frac{1}{K}$$

2. **Angle (Phase) Condition:**
   $$\angle G(s)H(s) = (2k + 1)180^\circ \quad (k = 0, \pm 1, \pm 2, \dots)$$

---

## 2. The Classical Evans Rules

`ctrlpy` automates the analytical derivation of all 7 classical rules:

### Rule 1: Number of Branches and Terminations
- The root locus has $n$ branches, where $n = \operatorname{deg}(D(s))$ is the number of open-loop poles.
- As $K$ increases from $0 \to \infty$, branches start at the open-loop poles ($K = 0$).
- $m$ branches terminate at the finite open-loop zeros ($K \to \infty$).
- The remaining $n - m$ branches terminate at infinity along asymptotes.

### Rule 2: Real-Axis Locus Segments
A point on the real axis belongs to the root locus if and only if the total number of real open-loop poles and zeros to its right is **odd**.

### Rule 3: Asymptote Centroid and Angles
The $n - m$ branches radiating toward infinity follow straight-line asymptotes intersecting the real axis at centroid $\sigma_a$:

$$\sigma_a = \frac{\sum_{i=1}^n \operatorname{Re}(p_i) - \sum_{j=1}^m \operatorname{Re}(z_j)}{n - m}$$

The angles of the asymptotes with the positive real axis are:

$$\theta_k = \frac{(2k + 1) \cdot 180^\circ}{n - m} \quad \text{for } k = 0, 1, \dots, n - m - 1$$

### Rule 4: Breakaway and Break-in Points
Points where locus branches depart from or arrive onto the real axis occur where the gain $K(s) = -\frac{D(s)}{N(s)}$ attains a local extremum:

$$\frac{dK}{ds} = 0 \iff D'(s)N(s) - D(s)N'(s) = 0$$

Roots of this equation that lie on the valid real-axis locus segments and yield $K \ge 0$ correspond to:
- **Breakaway points:** Local maxima of $K(s)$ on the real axis (branches depart into the complex plane).
- **Break-in points:** Local minima of $K(s)$ on the real axis (branches converge onto the real axis).

### Rule 5: Angles of Departure from Complex Poles
For a complex open-loop pole $p_i$, the angle of departure $\theta_{\text{dep}}$ is calculated from the angle condition:

$$\theta_{\text{dep}} = 180^\circ - \sum_{k \ne i} \angle(p_i - p_k) + \sum_{l=1}^m \angle(p_i - z_l)$$

### Rule 6: Angles of Arrival at Complex Zeros
For a complex open-loop zero $z_j$, the angle of arrival $\theta_{\text{arr}}$ is:

$$\theta_{\text{arr}} = 180^\circ - \sum_{l \ne j} \angle(z_j - z_l) + \sum_{k=1}^n \angle(z_j - p_k)$$

### Rule 7: Imaginary Axis Crossings ($j\omega$-axis)
Crossings with the imaginary axis indicate the onset of marginal stability. Substituting $s = j\omega$ into $D(s) + K N(s) = 0$ yields:

$$\operatorname{Re}(D(j\omega) + K N(j\omega)) = 0 \quad \text{and} \quad \operatorname{Im}(D(j\omega) + K N(j\omega)) = 0$$

Solving simultaneously provides the critical frequency $\omega_{\text{cross}}$ and critical gain $K_{\text{crit}}$.

---

## 3. Basic Python Usage

Import `RootLocusRules` from `ctrlpy.symbolic`:

```python
import ctrlpy as cp
from ctrlpy.symbolic import RootLocusRules

# Open-loop plant G(s) = 1 / (s(s+1)(s+2))
G = cp.tf([1], [1, 3, 2, 0])

# Derive all Evans rules analytically
rlr = RootLocusRules(G)

print(f"Number of branches: {rlr.num_branches}")
print(f"Asymptote centroid: sigma_a = {rlr.centroid}")
print(f"Asymptote angles: {rlr.asymptote_angles_deg}")
print(f"Breakaway points: {rlr.breakaway_points}")
print(f"j*omega crossings: {rlr.imag_axis_crossings}")
```

Output:
```text
Number of branches: 3
Asymptote centroid: sigma_a = -1.0
Asymptote angles: [60.0, 180.0, -60.0]
Breakaway points: [{'s': -0.4226, 'k': 0.3849, 'type': 'breakaway'}]
j*omega crossings: [{'omega': 1.4142, 'k': 6.0, 's': 1.4142j}]
```

---

## 4. Step-by-Step Pedagogical Derivations (`.explain_steps()`)

To generate detailed mathematical derivations for classroom handouts or verification:

```python
for step in rlr.explain_steps():
    print(f"- {step}")
```

---

## 5. Jupyter Notebook Rich LaTeX Rendering

Evaluating a `RootLocusRules` instance in a Jupyter notebook cell renders a structured summary table via LaTeX:

```python
rlr = RootLocusRules(G)
rlr  # Renders _repr_latex_()
```

Renders as:

$$\begin{aligned}
\textbf{Analytical Root Locus Derivation Summary:} & \\
\textbf{Rule 1 (Branches):} &\quad n = 3\text{ poles}, m = 0\text{ zeros} \implies 3\text{ branches}, 3\text{ asymptotes to }\infty \\
\textbf{Rule 2 (Real-Axis Segments):} &\quad s \in [-1, 0] \cup [-\infty, -2] \\
\textbf{Rule 3 (Asymptotes):} &\quad \sigma_a = -1, \quad \theta_k \in \{60^\circ, 180^\circ, -60^\circ\} \\
\textbf{Rule 4 (Breakaway/Break-in):} &\quad s = -0.4226 \text{ (breakaway, } K = 0.3849\text{)} \\
\textbf{Rule 5 (Departure Angles):} &\quad \text{No complex open-loop poles} \\
\textbf{Rule 6 (Arrival Angles):} &\quad \text{No complex open-loop zeros} \\
\textbf{Rule 7 (j\omega Crossings):} &\quad s = \pm j1.414 \text{ at } K_{\text{crit}} = 6
\end{aligned}$$

