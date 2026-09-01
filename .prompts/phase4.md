Phase 4: Implement Frequency-Domain Analysis, Stability Margins, and Visualization.

Project Standards:
- Target Python 3.10+ with strict type hinting.
- Follow PEP 8 standards with NumPy-style docstrings.
- Strict Separation of Concerns: Computation modules must NEVER import matplotlib. Visualization functions should only consume pre-computed data structures.

Tasks to complete:
1. Create `src/ctrlpy/freq_domain.py`:
   - Implement `bode_data(sys, omega=None)`:
     - Calculates frequencies `w` (rad/s), magnitude (linear and dB), and phase (degrees, unwrapped).
     - Auto-generates logarithmic frequency vector based on system poles/zeros if `omega` is None.
   - Implement `nyquist_data(sys, omega=None)`:
     - Returns complex frequency response $G(j\omega)$ avoiding division by zero at poles on imaginary axis.
   - Implement `root_locus_data(sys, gains=None)`:
     - Computes closed-loop pole trajectories for varying gains $k \ge 0$ using vectorized polynomial root-finding (`np.roots`) or state-space eigenvalue solvers (`scipy.linalg.eig`).
   - Implement `margin(sys) -> StabilityMargins`:
     - Dataclass returning Gain Margin (GM in dB), Phase Margin (PM in degrees), Gain Crossover Frequency (Wcg), and Phase Crossover Frequency (Wcp).

2. Create `src/ctrlpy/plotting.py`:
   - Implement `plot_bode(sys, omega=None, margins=True, ax=None)`:
     - Two subplots (Magnitude in dB vs $\log\omega$, Phase in deg vs $\log\omega$) with grid lines and optional margin annotations.
   - Implement `plot_nyquist(sys, omega=None, ax=None)`:
     - Plots $\text{Re}(G(j\omega))$ vs $\text{Im}(G(j\omega))$, marks critical point $(-1, 0j)$, and includes unit circle.
   - Implement `plot_root_locus(sys, gains=None, ax=None)`:
     - Plots root trajectories in complex s-plane with 'x' for open-loop poles and 'o' for zeros.
   - Implement `plot_step(sys, T=None, ax=None)`:
     - Quick helper for time-domain step plot.
   - All plotting functions must return the Matplotlib `Figure` and `Axes` objects and accept an optional existing `ax`.

3. Update `src/ctrlpy/__init__.py`:
   - Export all new frequency computation and plotting routines.

4. Create unit tests in `tests/test_freq_domain.py`:
   - Test Bode calculation of simple integrator $1/s$: slope must be $-20\text{ dB/decade}$ and phase $-90^\circ$.
   - Test stability margins of standard second-order system against known analytical values.
   - Test Root Locus open-loop endpoints (start at poles when $k=0$, approach zeros or asymptotes as $k \to \infty$).
   - Test that plotting functions run without errors and return valid Matplotlib Figure/Axes instances (use headless backend `matplotlib.use('Agg')` in tests).

Verification Step:
- Run `uv run ruff check .` and `uv run ruff format --check .`
- Run `uv run pytest` to ensure 100% test pass rate across all project tests.