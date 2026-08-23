Phase 3: Implement Time-Domain Simulation and Response Functions.

Project Standards:
- Target Python 3.10+ with strict type hinting.
- Follow PEP 8 standards with NumPy-style docstrings.
- Core simulation logic should wrap and optimize SciPy numerical solvers (`scipy.signal.step`, `scipy.signal.impulse`, `scipy.signal.lsim`).

Tasks to complete:
1. Create `src/ctrlpy/simulation_results.py`:
   - Define a dataclass `TimeResponseData` with attributes:
     - `t`: 1D NumPy float array (time vector).
     - `y`: 1D or 2D NumPy float array (system output).
     - `x`: Optional 2D NumPy float array (state trajectories, if StateSpace).
   - Implement properties/methods on `TimeResponseData` to calculate key metrics:
     - `steady_state_value()`
     - `rise_time()` (10% to 90%)
     - `settling_time(tolerance=0.02)` (2% standard)
     - `overshoot()` (peak percentage above steady-state)
     - `peak_time()`

2. Create `src/ctrlpy/time_domain.py`:
   - Implement `step_response(sys, T=None, X0=None) -> TimeResponseData`:
     - Accepts `TransferFunction` or `StateSpace`.
     - Automatically generates an appropriate time horizon `T` if `None` based on system dominant poles.
   - Implement `impulse_response(sys, T=None, X0=None) -> TimeResponseData`.
   - Implement `forced_response(sys, T, U, X0=None) -> TimeResponseData` for arbitrary inputs `U(t)`.
   - Add convenience wrapper methods `.step()` and `.impulse()` directly onto the `LinearTimeInvariant` base class.

3. Update `src/ctrlpy/__init__.py`:
   - Export `step_response`, `impulse_response`, `forced_response`, and `TimeResponseData`.

4. Create unit tests in `tests/test_time_domain.py`:
   - Test first-order system $G(s) = \frac{1}{\tau s + 1}$ step response: verify $y(\tau) \approx 0.632$ and steady-state $= 1.0$.
   - Test standard second-order underdamped system $G(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}$: verify analytical percent overshoot and peak time formulas.
   - Test equivalence of step/impulse responses between `TransferFunction` and its equivalent `StateSpace` conversion.
   - Test forced response with sinusoidal and ramp inputs.

Verification Step:
- Run `uv run ruff check .` and `uv run ruff format --check .`
- Run `uv run pytest` to ensure 100% test pass rate across all modules.