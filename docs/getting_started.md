# Getting Started with `ctrlpy`

This guide walks you through modeling Linear Time-Invariant (LTI) systems, connecting blocks in series/parallel/feedback, simulating transient dynamics, computing frequency responses and stability margins, and visualizing results.

---

## 1. Creating LTI Models

`ctrlpy` supports continuous-time Transfer Functions and State-Space representations.

### Transfer Functions (`TransferFunction`, `tf`)

A single-input single-output (SISO) transfer function:

$$G(s) = \frac{N(s)}{D(s)} = \frac{b_m s^m + \dots + b_0}{a_n s^n + \dots + a_0}$$

```python
import ctrlpy as cp

# G(s) = (2s + 5) / (s^2 + 3s + 2)
G = cp.tf([2, 5], [1, 3, 2])

print("Transfer Function:")
print(G)
print(f"Poles: {G.poles()}")
print(f"Zeros: {G.zeros()}")
```

### State-Space Models (`StateSpace`, `ss`)

Continuous-time state-space systems:

$$\dot{x}(t) = A x(t) + B u(t), \quad y(t) = C x(t) + D u(t)$$

```python
import numpy as np
import ctrlpy as cp

A = [[-3.0, -2.0], [1.0, 0.0]]
B = [[1.0], [0.0]]
C = [[0.0, 10.0]]
D = [[0.0]]

sys = cp.ss(A, B, C, D)
print("State Space Model:")
print(sys)
```

### Model Conversions

```python
# Convert TransferFunction to StateSpace
sys_ss = G.to_ss()

# Convert StateSpace to TransferFunction
sys_tf = sys.to_tf()
```

---

## 2. Block Diagram Algebra & Interconnections

Combine systems using standard arithmetic operators (`+`, `-`, `*`, `/`) or dedicated functions (`series`, `parallel`, `feedback`):

```python
import ctrlpy as cp

G1 = cp.tf([1], [1, 2])
G2 = cp.tf([3], [1, 4])

# Series (Cascade) Connection: G_series = G2 * G1
G_ser = cp.series(G1, G2)

# Parallel Connection: G_parallel = G1 + G2
G_par = cp.parallel(G1, G2)

# Closed-Loop Feedback Connection: T = G1 / (1 + G1 * G2)
T_closed = cp.feedback(G1, G2, sign=-1)
```

---

## 3. Time-Domain Simulations & Metric Extraction

Simulate step, impulse, and forced responses:

```python
import ctrlpy as cp

# Define closed-loop system
T = cp.tf([4], [1, 2, 4])

# Step response
resp = cp.step_response(T, T=8.0)

print(f"Steady-State Value : {resp.steady_state_value():.4f}")
print(f"Rise Time (10%-90%): {resp.rise_time():.4f} s")
print(f"Settling Time (2%) : {resp.settling_time(tolerance=0.02):.4f} s")
print(f"Percent Overshoot  : {resp.overshoot():.2f} %")
print(f"Peak Time          : {resp.peak_time():.4f} s")
```

---

## 4. Frequency-Domain & Stability Margins

Compute Bode, Nyquist, and Root Locus responses:

```python
import ctrlpy as cp

# Open-loop system
L = cp.tf([4], [1, 3, 2, 0])

# Exact stability margins
sm = cp.margin(L)
print(f"Gain Margin (GM)           : {sm.gm_db:.2f} dB")
print(f"Phase Margin (PM)          : {sm.pm_deg:.2f}°")
print(f"Gain Crossover Freq (wcg)  : {sm.wcg:.4f} rad/s")
print(f"Phase Crossover Freq (wcp) : {sm.wcp:.4f} rad/s")
```

---

## 5. Dual-Backend Visualizations

`ctrlpy` provides dual visualization backends: static Matplotlib plots and interactive Plotly charts.

### Static Plotting (Matplotlib)

```python
import matplotlib.pyplot as plt
import ctrlpy as cp

L = cp.tf([10], [1, 3, 2, 0])
T = cp.feedback(L, 1.0)

fig1, ax1 = cp.plot_step(T)
fig2, (ax_mag, ax_phase) = cp.plot_bode(L, margins=True)
fig3, ax3 = cp.plot_nyquist(L)
fig4, ax4 = cp.plot_root_locus(L)

plt.show()
```

### Interactive Visualizations (Plotly)

```python
import ctrlpy as cp

L = cp.tf([10], [1, 3, 2, 0])
T = cp.feedback(L, 1.0)

fig_step = cp.iplot_step(T)
fig_bode = cp.iplot_bode(L, margins=True)
fig_nyquist = cp.iplot_nyquist(L)
fig_rlocus = cp.iplot_root_locus(L)

# Display interactive figures
fig_step.show()
```

---

## 6. Pedagogical & Educational Derivations (`ctrlpy.pedagogy`)

When `ctrlpy[symbolic]` is installed, access step-by-step mathematical derivations:

```python
import ctrlpy as cp
from ctrlpy.pedagogy import root_locus_rules, routh_table, steady_state_analysis

G = cp.tf([1], [1, 3, 2, 0])

# 1. Routh-Hurwitz Stability Criterion & Gain Margin Range
routh_res = routh_table(G, k_symbol="K")
print(routh_res)
print(f"Stable K Range: {routh_res.k_range}")

# 2. Analytical Evans Root Locus Rules
rl_res = root_locus_rules(G)
print(rl_res)

# 3. System Type & Steady-State Tracking Errors
ss_res = steady_state_analysis(G)
print(ss_res)
```

