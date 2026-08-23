Phase 6: Comprehensive DX Upgrade - PID Controller Design, Fluent Plotting API, and Interactive Plotly Support.

Project Standards:
- Target Python 3.10+ with strict type hints.
- Clean separation: Mathematical and simulation logic remains independent of visualization backends.
- Follow PEP 8 formatting with NumPy docstrings.

Tasks to complete:

1. Update Dependencies in `pyproject.toml`:
   - Add `plotly>=5.15.0` to runtime dependencies.
   - Run `uv sync` to ensure dependencies are installed.

2. Implement PID Controller Module (`src/ctrlpy/controllers.py`):
   - Implement `PID(Kp=1.0, Ki=0.0, Kd=0.0, Tf=0.0, N=None)`:
     - Returns a `TransferFunction` representing standard PID control with optional derivative filter:
       C(s) = Kp + Ki/s + (Kd * s) / (Tf * s + 1)
       (or using derivative filter coefficient N where Tf = Kd / (N * Kp)).
     - If Ki == 0 and Kd == 0, return pure proportional gain.
     - Ensure proper polynomial simplification so the resulting transfer function is proper.
   - Implement convenience functions: `pid(Kp, Ki, Kd, ...)`, `pi(Kp, Ki)`, `pd(Kp, Kd)`.
   - Implement tuning heuristic:
     - `tune_ziegler_nichols(sys, method="step")`: returns configured PID instance based on step response characteristics (delay L, slope R).
   - Export PID constructors and helpers in `src/ctrlpy/__init__.py`.

3. Implement Plotly Interactive Plotting Engine (`src/ctrlpy/plotting_plotly.py`):
   - Implement `iplot_bode(sys, omega=None, margins=True) -> go.Figure`:
     - Stacked subplots (Magnitude in dB vs log frequency, Phase in deg vs log frequency).
     - Hover template displaying Frequency (rad/s), Gain (dB), and Phase (deg).
     - Visual indicators for Gain/Phase crossover frequencies (Wcg, Wcp).
   - Implement `iplot_nyquist(sys, omega=None) -> go.Figure`:
     - Parametric complex trajectory with hover showing omega, Re, Im.
     - Red '+' marker at critical point (-1, 0) and dashed unit circle.
     - Includes positive and negative frequency branches.
   - Implement `iplot_root_locus(sys, gains=None) -> go.Figure`:
     - Complex s-plane trajectory.
     - Hover template displaying gain K, pole location, damping ratio zeta, and natural frequency wn.
     - Open-loop poles marked with 'x', zeros marked with 'o'.
   - Implement `iplot_step(sys, T=None) -> go.Figure`:
     - Interactive step response with markers for steady-state value, rise time, and peak overshoot.

4. Integrate Fluent Plotting API into `LinearTimeInvariant` Base Class (`src/ctrlpy/models/base.py`):
   - Add instance methods directly to the base class (inherited by TransferFunction and StateSpace):
     - `sys.plot_step(backend="matplotlib", ...)` / `sys.iplot_step(...)`
     - `sys.plot_impulse(backend="matplotlib", ...)` / `sys.iplot_impulse(...)`
     - `sys.plot_bode(backend="matplotlib", ...)` / `sys.iplot_bode(...)`
     - `sys.plot_nyquist(backend="matplotlib", ...)` / `sys.iplot_nyquist(...)`
     - `sys.plot_root_locus(backend="matplotlib", ...)` / `sys.iplot_root_locus(...)`
   - If backend is "matplotlib", delegate to `src/ctrlpy/plotting.py` and handle `plt.show()` / `(fig, ax)` return cleanly.
   - If backend is "plotly" (or via `iplot_*`), delegate to `src/ctrlpy/plotting_plotly.py` and return `go.Figure`.

5. Create Comprehensive Case Study Notebook (`notebooks/pid_and_interactive_analysis.ipynb`):
   - Build an end-to-end tutorial demonstrating:
     1. Setting up an uncompensated 3rd order plant: G(s) = 1 / ((s + 1)(s + 2)(s + 5)).
     2. Analyzing baseline uncompensated performance with fluent interactive methods: `sys.iplot_step()` and `sys.iplot_bode()`.
     3. Designing a PID controller using `pid(Kp, Ki, Kd, Tf=...)` and tuning via `tune_ziegler_nichols`.
     4. Interactive Root Locus tuning with `sys.iplot_root_locus()`, inspecting damping and pole trajectories.
     5. Comparing closed-loop step responses before and after PID compensation on a single figure.

6. Unit Tests and Quality Verification:
   - Create `tests/test_controllers.py` testing PID algebraic forms, filter cutoff behavior, and tuning logic.
   - Create `tests/test_plotting_plotly.py` testing that Plotly functions return valid `go.Figure` instances with correct trace structures.
   - Test base class fluent plotting methods for both backends.
   - Run `uv run ruff format .` and `uv run ruff check .`
   - Run `uv run mypy .`
   - Run `uv run pytest` (100% pass rate).