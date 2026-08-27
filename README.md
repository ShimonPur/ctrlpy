# ctrlpy

[![CI](https://github.com/ShimonPur/ctrlpy/actions/workflows/ci.yml/badge.svg)](https://github.com/ShimonPur/ctrlpy/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Coverage](https://img.shields.io/badge/coverage-%3E90%25-brightgreen.svg)](https://pytest-cov.readthedocs.io/)

**A modern, high-performance Python control systems library built on NumPy and SciPy.**

`ctrlpy` provides an intuitive, fully type-annotated, and fast framework for classical and modern control systems modeling, simulation, frequency-domain analysis, and interactive visualization.

---

## Key Features

- **Intuitive LTI System Modeling**:
  - Continuous-time **Transfer Functions** (`TransferFunction`, `tf`) with polynomial arithmetic and pole/zero calculation.
  - Continuous-time **State-Space** models (`StateSpace`, `ss`) supporting arbitrary state dimensions ($A, B, C, D$).
  - Seamless bidirectional model conversions (`tf.to_ss()`, `ss.to_tf()`).
- **Algebraic Block Diagram Arithmetic**:
  - Natural operator overloading (`+`, `-`, `*`, `/`) for system combinations.
  - High-level interconnection functions: `series()`, `parallel()`, and `feedback()` (unity, non-unity, negative, and positive feedback loops).
- **Comprehensive Time-Domain Simulations**:
  - Step (`step_response`), Impulse (`impulse_response`), and Arbitrary Input (`forced_response`) simulation routines.
  - Rich `TimeResponseData` container with automated interpolation for high-precision metric extraction:
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
  - Native LaTeX formatting (`_repr_latex_`) rendering mathematical fractions, Routh tables, and analytical derivation summaries directly in notebook cells.
- **Modern Python Standards**:
  - 100% Type-annotated codebase with strict `mypy` validation (`py.typed` included).
  - Clean PEP 8 formatting with `ruff`.

---

## Project Structure

```text
ctrlpy/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI matrix (Python 3.10, 3.11, 3.12)
├── notebooks/
│   ├── 01_quickstart.ipynb                     # LTI modeling, interconnections, and plotting
│   ├── 02_dc_motor_control.ipynb               # DC motor speed control & PI design
│   ├── 03_mass_spring_damper.ipynb             # Mechanical oscillator & Root Locus PID tuning
│   ├── 04_frequency_domain_deep_dive.ipynb     # Bode margins, Nyquist criteria & indented contours
│   └── 05_pedagogical_and_symbolic_tools.ipynb # Routh tables, Root Locus rules, and steady-state error
├── src/
│   └── ctrlpy/
│       ├── __init__.py        # Root public API exports (Pure numerical)
│       ├── algebra.py         # Block diagram algebra (series, parallel, feedback)
│       ├── controllers.py     # PID/PI/PD controllers & Ziegler-Nichols tuning
│       ├── exceptions.py      # Custom exception hierarchy
│       ├── freq_domain.py     # Frequency responses & exact stability margins
│       ├── models/            # LTI base class, TransferFunction, and StateSpace
│       ├── pedagogy/          # Optional symbolic & pedagogical module (SymPy)
│       │   ├── __init__.py    # Submodule exports & defensive import guard
│       │   ├── root_locus_rules.py # Analytical Evans Root Locus rules
│       │   ├── routh.py       # Routh-Hurwitz criterion & parametric K-solver
│       │   └── steady_state.py # System Type & static error constants
│       ├── plotting.py        # Static Matplotlib plotting routines
│       ├── plotting_plotly.py # Interactive Plotly visualization engine
│       ├── py.typed           # PEP 561 typing marker
│       ├── simulation_results.py # Typed TimeResponseData container & metric extraction
│       └── time_domain.py     # Vectorized step, impulse, and forced response solvers
├── tests/                     # Comprehensive test suite (>90% coverage)
├── pyproject.toml             # Packaging metadata, dependencies, and tool configs
└── README.md
```

---

## Installation

`ctrlpy` features an isolated architecture: the core library is 100% numerical and lightweight (NumPy, SciPy, Matplotlib, Plotly). Symbolic educational tools are available via an optional `[symbolic]` extra.

### Base Installation (Pure Numerical Engine)

Using `uv`:
```bash
uv add git+https://github.com/ShimonPur/ctrlpy.git
```

Using `pip`:
```bash
pip install git+https://github.com/ShimonPur/ctrlpy.git
```

### With Educational / Symbolic Submodule (`ctrlpy.pedagogy`)

To install `ctrlpy` along with `sympy` for step-by-step pedagogical derivations:

Using `uv`:
```bash
uv add git+https://github.com/ShimonPur/ctrlpy.git --extra symbolic
```

Using `pip`:
```bash
pip install "ctrlpy[symbolic]@git+https://github.com/ShimonPur/ctrlpy.git"
```

### Development & All Extras

```bash
# Using uv (editable with all extras)
uv sync --all-extras

# Using pip
pip install -e ".[symbolic,dev]"
```

---

## Quickstart

Here is a complete copy-paste example demonstrating model creation, closed-loop feedback, step response metric extraction, and Bode plotting with stability margin annotations:

```python
import matplotlib.pyplot as plt
import numpy as np

import ctrlpy as cp

# 1. Define Plant and PI Controller Transfer Functions
# Plant: G(s) = 1 / (s^2 + 2s + 1)
G_plant = cp.tf([1], [1, 2, 1])

# PI Controller: C(s) = (2s + 3) / s
G_ctrl = cp.tf([2, 3], [1, 0])

# 2. Form Open-Loop and Closed-Loop Systems
# Series cascade: G_open(s) = C(s) * G(s)
G_open = cp.series(G_ctrl, G_plant)

# Closed-loop unity feedback: T(s) = G_open / (1 + G_open)
T_closed = cp.feedback(G_open, 1)

print("Closed-Loop Transfer Function:")
print(T_closed)
print(f"Closed-Loop Poles: {T_closed.poles()}")

# 3. Simulate Time-Domain Step Response & Extract Dynamic Metrics
resp = cp.step_response(T_closed, T=8.0)

print("\n--- Transient Response Metrics ---")
print(f"Steady-State Value : {resp.steady_state_value():.4f}")
print(f"Rise Time (10%-90%): {resp.rise_time():.4f} s")
print(f"Settling Time (2%) : {resp.settling_time(tolerance=0.02):.4f} s")
print(f"Percent Overshoot  : {resp.overshoot():.2f} %")
print(f"Peak Time          : {resp.peak_time():.4f} s")

# 4. Compute Exact Stability Margins
sm = cp.margin(G_open)
print("\n--- Frequency Stability Margins ---")
print(f"Gain Margin (GM)           : {sm.gm_db:.2f} dB")
print(f"Phase Margin (PM)          : {sm.pm_deg:.2f}°")
print(f"Gain Crossover Freq (wcg)  : {sm.wcg:.3f} rad/s")
print(f"Phase Crossover Freq (wcp) : {sm.wcp:.3f} rad/s")

# 5. Visualize Step Response and Bode Diagram
fig1, ax1 = cp.plot_step(T_closed, T=8.0)
fig2, (ax_mag, ax_phase) = cp.plot_bode(G_open, margins=True)

plt.show()
```

### Educational & Symbolic Quickstart (`ctrlpy.pedagogy`)

When `ctrlpy[symbolic]` is installed, access step-by-step mathematical derivations:

```python
import ctrlpy as cp
from ctrlpy.pedagogy import root_locus_rules, routh_table, steady_state_analysis

# 1. Step-by-Step Routh-Hurwitz Array & Parametric Gain Bounds
G_plant = cp.tf([1], [1, 3, 2, 0])
routh_res = routh_table(G_plant, k_symbol="K")
print(routh_res)
print(f"Stable Gain Margin: {routh_res.k_range}")

# 2. Analytical Evans Root Locus Rules
rl_res = root_locus_rules(G_plant)
print(rl_res)

# 3. System Type & Steady-State Error Constants
ss_res = steady_state_analysis(G_plant)
print(ss_res)
```

---

## Practical Engineering Case Studies

Explore the hands-on tutorial and engineering case studies located in [`notebooks/`](notebooks/):

1. **[01. Quickstart Guide](notebooks/01_quickstart.ipynb)**:
   - Fundamentals of LTI modeling (`TransferFunction`, `StateSpace`).
   - Bidirectional conversions, polynomial arithmetic, and block diagram algebra (`series`, `parallel`, `feedback`).
   - Time-domain simulation, metric interpolation, and static/interactive plotting.

2. **[02. DC Motor Speed Control & PI Design](notebooks/02_dc_motor_control.ipynb)**:
   - First-principles electromechanical modeling ($J, b, K_t, K_e, R, L$).
   - Analysis of open-loop steady-state tracking error ($e_{ss}$).
   - PI controller synthesis, step tracking performance, and load torque disturbance rejection ($\Delta \tau_L$).

3. **[03. Mass-Spring-Damper & Root Locus PID Tuning](notebooks/03_mass_spring_damper.ipynb)**:
   - Mechanical 2nd-order oscillator dynamics ($\omega_n, \zeta$).
   - Root Locus parameter sweeps and pole migration analysis with Plotly.
   - PID tuning to achieve exact damping ($\zeta \ge 0.707$), rapid settling ($t_s \le 1.5\text{ s}$), and zero steady-state error.

4. **[04. Frequency-Domain Deep Dive & Nyquist Analysis](notebooks/04_frequency_domain_deep_dive.ipynb)**:
   - Bode stability margins ($\mathrm{GM}, \mathrm{PM}, \omega_{cg}, \omega_{cp}$) across subcritical, critical, and supercritical gain regimes.
   - Cauchy's Argument Principle and Nyquist stability criterion ($Z = N + P$).
   - Indented contour mapping for open-loop poles on the imaginary axis (integrator $s=0$).

5. **[05. Pedagogical & Symbolic Control Tools](notebooks/05_pedagogical_and_symbolic_tools.ipynb)**:
   - Step-by-step Routh-Hurwitz arrays with $\epsilon > 0$ substitution and auxiliary polynomial $A(s)$ resolution.
   - Closed-loop parametric stability range ($K_{\min} < K < K_{\max}$) solving.
   - Classroom analytical Evans Root Locus rules (asymptotes, centroid, breakaway/break-in, departure angles, $j\omega$ crossings).
   - System Type ($0, 1, 2$) classification, static constants ($K_p, K_v, K_a$), and steady-state error tracking.

---

## Comparison: `ctrlpy` vs Legacy `python-control`

| Feature | `ctrlpy` | Legacy `python-control` |
| :--- | :--- | :--- |
| **Type Safety** | 100% Strict Type Annotations & `py.typed` | Partial / Legacy docstring types |
| **Modern Data Structures** | Rich typed Dataclasses (`TimeResponseData`, `StabilityMargins`, `BodeData`) | Raw tuples or untyped custom objects |
| **Performance** | Fully vectorized NumPy & SciPy numerical routines | Mixed legacy code and wrapper overhead |
| **Metric Extraction** | Built-in sub-interval linear interpolation ($t_r, t_s, \%OS, t_p$) | Basic indexing or external utility functions |
| **Interactive Plotting** | Built-in Plotly interactive charts (`iplot_step`, `iplot_bode`, etc.) | Matplotlib only by default |
| **Jupyter Integration** | Native mathematical LaTeX formatting (`_repr_latex_`) | Plain text representations |
| **Build & Packaging** | Modern `pyproject.toml` (PEP 517 / PEP 621 / `hatchling` / `uv`) | Legacy `setup.py` / `setuptools` |
| **Simplicity & Weight** | Lightweight, zero C/Fortran compile hurdles | External Slycot dependency required for certain state-space operations |

---

## Quality Checks & Testing

`ctrlpy` maintains a comprehensive test suite with >90% code coverage and strict linting standards:

```bash
# Run test suite with coverage report
uv run pytest

# Check strict static types
uv run mypy .

# Check code formatting & linting
uv run ruff check .
uv run ruff format --check .

# Test package distribution build
uv build
```

---

## License

`ctrlpy` is distributed under the terms of the [MIT License](LICENSE).

