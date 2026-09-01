# ctrlpy

<div align="center">
  <p><strong>A modern, high-performance Python control systems library built on NumPy, SciPy, and Plotly.</strong></p>
  <p>
    <a href="https://pypi.org/project/ctrlpy/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python Version"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Code Style: Ruff"></a>
    <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/type%20checked-mypy-blue.svg" alt="Type Checked: mypy"></a>
    <a href="https://pytest-cov.readthedocs.io/"><img src="https://img.shields.io/badge/coverage-%3E90%25-brightgreen.svg" alt="Coverage"></a>
  </p>
</div>

---

`ctrlpy` provides an intuitive, strictly type-annotated, and fast framework for classical and modern control systems engineering, time/frequency-domain analysis, pedagogical derivations, and interactive visualizations.

## Key Capabilities

- **Intuitive LTI System Modeling**:
    - Continuous-time **Transfer Functions** (`TransferFunction`, `tf`) with polynomial arithmetic and pole/zero calculation.
    - Continuous-time **State-Space** models (`StateSpace`, `ss`) supporting arbitrary state dimensions ($A, B, C, D$).
    - Seamless bidirectional model conversions (`tf.to_ss()`, `ss.to_tf()`).
- **Algebraic Block Diagram Arithmetic**:
    - Natural operator overloading (`+`, `-`, `*`, `/`) for system interconnections.
    - High-level interconnection functions: `series()`, `parallel()`, and `feedback()` (unity, non-unity, negative, and positive feedback loops).
- **Comprehensive Time-Domain Simulations**:
    - Step (`step_response`), Impulse (`impulse_response`), and Arbitrary Input (`forced_response`) simulation routines.
    - Rich `TimeResponseData` container with automated sub-sample interpolation for high-precision metric extraction:
        - Rise time $t_r$ (10% to 90%)
        - Settling time $t_s$ (arbitrary error band, e.g. 2%)
        - Percent overshoot $\%OS$ and peak time $t_p$
        - Steady-state value $y_{ss}$
- **Frequency-Domain & Stability Analysis**:
    - Vectorized frequency responses (`bode_data`, `nyquist_data`, `root_locus_data`).
    - Exact stability margins computation (`margin`): Gain Margin ($\mathrm{GM}$), Phase Margin ($\mathrm{PM}$), Gain Crossover Frequency ($\omega_{cg}$), and Phase Crossover Frequency ($\omega_{cp}$).
    - Diagnostic plotting routines (`plot_bode`, `plot_nyquist`, `plot_root_locus`, `plot_step`).
    - Interactive Plotly visualizations (`iplot_bode`, `iplot_nyquist`, `iplot_root_locus`, `iplot_step`).
- **Pedagogical & Educational Derivations (`ctrlpy.pedagogy`)**:
    - Isolated submodule for step-by-step classroom analysis without overhead in the core numerical engine.
    - **Routh-Hurwitz Stability Criterion** (`routh_table`): Full array construction, $\epsilon > 0$ substitution for leading zeros, auxiliary polynomial $A(s)$ resolution for rows of zeros, and parametric gain range ($K_{\min} < K < K_{\max}$) solver.
    - **Classroom Root Locus Rules** (`root_locus_rules`): Complete step-by-step derivation of branches, real-axis segments, asymptotes ($\sigma_a, \theta_k$), breakaway/break-in points ($dK/ds = 0$), departure/arrival angles, and $j\omega$-crossings.
    - **Steady-State Error Analysis** (`steady_state_analysis`): Automatic System Type ($0, 1, 2\dots$) classification, static constants ($K_p, K_v, K_a$), steady-state errors ($e_{ss}$ for step, ramp, parabola), and closed-loop stability verification.
- **Jupyter Notebook Integration**:
    - Native LaTeX formatting (`_repr_latex_` and `_repr_markdown_`) rendering mathematical fractions, Routh tables, and analytical derivation summaries directly in notebook cells.

---

## Installation

`ctrlpy` features an isolated architecture: the core library is 100% numerical and lightweight (NumPy, SciPy, Matplotlib, Plotly). Symbolic educational tools are available via an optional `[symbolic]` extra.

### Base Installation (Pure Numerical Engine)

=== "uv"
    ```bash
    uv add git+https://github.com/ShimonPur/ctrlpy.git
    ```

=== "pip"
    ```bash
    pip install git+https://github.com/ShimonPur/ctrlpy.git
    ```

### With Educational / Symbolic Submodule (`ctrlpy.pedagogy`)

=== "uv"
    ```bash
    uv add git+https://github.com/ShimonPur/ctrlpy.git --extra symbolic
    ```

=== "pip"
    ```bash
    pip install "ctrlpy[symbolic]@git+https://github.com/ShimonPur/ctrlpy.git"
    ```

### Development & All Extras

=== "uv"
    ```bash
    uv sync --all-extras
    ```

=== "pip"
    ```bash
    pip install -e ".[symbolic,docs,dev]"
    ```

---

## Quick Example

```python
import ctrlpy as cp
import matplotlib.pyplot as plt

# 1. Define 2nd-order plant: G(s) = 10 / (s^2 + 3s + 2)
G = cp.tf([10], [1, 3, 2])

# 2. Design a PI controller: C(s) = (2s + 1) / s
C = cp.pi(Kp=2.0, Ki=1.0)

# 3. Create closed-loop system: T(s) = C*G / (1 + C*G)
L = cp.series(C, G)
T = cp.feedback(L, 1.0)

# 4. Simulate step response & extract performance metrics
resp = cp.step_response(T, T=6.0)
print(f"Rise Time (10%-90%): {resp.rise_time():.4f} s")
print(f"Settling Time (2%) : {resp.settling_time():.4f} s")
print(f"Percent Overshoot  : {resp.overshoot():.2f} %")

# 5. Frequency-domain stability margins
sm = cp.margin(L)
print(f"Gain Margin : {sm.gm_db:.2f} dB")
print(f"Phase Margin: {sm.pm_deg:.2f}°")

# 6. Plot static or interactive charts
cp.plot_step(T)
cp.plot_bode(L, margins=True)
plt.show()
```

---

## Interactive Visualizations with Plotly

`ctrlpy` includes native Plotly interactive plotting out of the box:

```python
# Open interactive Plotly figures in your browser or Jupyter notebook
fig_step = cp.iplot_step(T)
fig_bode = cp.iplot_bode(L)
fig_nyquist = cp.iplot_nyquist(L)
fig_rlocus = cp.iplot_root_locus(G)
```

