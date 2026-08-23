Phase 8: Isolated Pedagogical & Symbolic Submodule (`ctrlpy.pedagogy`)

Architecture Standards:
- Core library remains 100% numerical, fast, and independent of symbolic engines.
- Create an isolated submodule `src/ctrlpy/pedagogy/` for step-by-step educational analysis.
- `sympy` is configured strictly as an optional dependency (`symbolic` group in `pyproject.toml`).
- Include rich LaTeX rendering (`_repr_latex_`) for seamless, formatted display in Jupyter notebooks.
- Maintain full type safety and clean PEP 8 standards with NumPy docstrings.

Tasks to complete:

1. Update `pyproject.toml` Configuration:
   - Add optional dependency group under `[project.optional-dependencies]`:
     `symbolic = ["sympy>=1.12.0"]`.
   - Update `dev` dependency group to include `sympy>=1.12.0` to enable test execution.
   - Run `uv sync --extra symbolic` to install optional dependencies in the virtual environment.

2. Implement Routh-Hurwitz Stability Criterion (`src/ctrlpy/pedagogy/routh.py`):
   - Implement `routh_table(poly_or_sys, variable="s", k_symbol=None)`:
     - Accepts numerical coefficient lists, TransferFunction instances, or SymPy symbolic expressions.
     - Constructs the complete Routh array step-by-step.
     - Handles textbook special cases:
       1. Zero in the first column: replaces with symbolic epsilon $\epsilon > 0$ and evaluates polynomial sign changes in the limit $\epsilon \to 0^+$.
       2. Row of all zeros: constructs auxiliary polynomial $A(s)$, computes $\frac{dA}{ds}$, and completes the array to detect symmetric / imaginary-axis poles.
     - Stability analysis: counts sign changes in column 1 to report the exact number of Right-Half Plane (RHP) poles.
     - Parametric K-stability solver: when a symbolic gain $K$ is provided, evaluates column 1 inequalities to return the valid stability range $K_{\min} < K < K_{\max}$.
   - Create `RouthResult` dataclass:
     - Fields: `table` (SymPy matrix / list of rows), `num_rhp_poles: int`, `is_stable: bool`, `k_range: str | None`, `steps: list[str]`.
     - Implement `_repr_latex_()` rendering a formatted LaTeX matrix/table with highlighted first-column sign variations.

3. Implement Analytical Root Locus Derivations (`src/ctrlpy/pedagogy/root_locus_rules.py`):
   - Implement `root_locus_rules(sys)`:
     - Calculates formal classroom analytical rules:
       - Number of open-loop poles ($n$), zeros ($m$), and branches to infinity ($n - m$).
       - Real-axis root locus segments.
       - Asymptote angles $\theta_k = \frac{(2k+1)\cdot 180^\circ}{n-m}$ and centroid $\sigma_a = \frac{\sum p_i - \sum z_i}{n-m}$.
       - Breakaway and break-in points via analytic solution of $\frac{dK}{ds} = 0$.
       - Departure angles from complex poles and arrival angles to complex zeros.
       - Imaginary axis crossing points ($s = j\omega$) and critical gains $K_{\text{crit}}$.
     - Return `RootLocusRulesResult` with LaTeX formatting summarizing all steps.

4. Implement Steady-State Error Analysis (`src/ctrlpy/pedagogy/steady_state.py`):
   - Implement `steady_state_analysis(sys)`:
     - Automatically identifies System Type (0, 1, 2...).
     - Computes static error constants: $K_p$ (Position), $K_v$ (Velocity), $K_a$ (Acceleration).
     - Computes steady-state errors $e_{ss}$ for unit step, ramp, and parabolic inputs.
     - Return `SteadyStateResult` with LaTeX table representation.

5. Expose Submodule API cleanly (`src/ctrlpy/pedagogy/__init__.py`):
   - Export: `routh_table`, `root_locus_rules`, `steady_state_analysis`, `RouthResult`, `RootLocusRulesResult`, `SteadyStateResult`.
   - Implement a defensive guard: if `sympy` is not installed, raise an informative `ImportError` directing the user to install `ctrlpy[symbolic]`.

6. Create Educational Jupyter Notebook (`notebooks/05_pedagogical_and_symbolic_tools.ipynb`):
   - Structured classroom-style guide demonstrating:
     1. Constructing step-by-step Routh tables for 3rd and 4th order transfer functions.
     2. Solving special cases: epsilon substitution and all-zero rows for marginal stability.
     3. Finding closed-loop parametric gain bounds ($K$-range).
     4. Deriving Root Locus asymptotes, breakaway points, and departure angles.
     5. System type identification and static error constants verification.

7. Update `README.md`:
   - Document the architectural split: Core numerical engine vs. Optional pedagogical submodule.
   - Document explicit installation commands:
     * Base installation (Numerical only): `pip install ctrlpy` or `uv add ctrlpy`
     * With Educational/Symbolic extension: `pip install "ctrlpy[symbolic]"` or `uv add ctrlpy --extra symbolic`
     * Development / All extras: `pip install -e ".[symbolic,dev]"` or `uv sync --all-extras`
   - Add a quickstart code snippet demonstrating `from ctrlpy.pedagogy import routh_table`.

8. Unit Tests and Verification:
   - Create `tests/test_pedagogy.py`:
     - Test standard Routh tables against known textbook polynomials.
     - Test epsilon substitution and all-zero auxiliary polynomial resolution.
     - Test parametric $K$-range solver.
     - Test Root Locus rule derivations and steady-state error calculations.
   - Run `uv run ruff format .` and `uv run ruff check .`
   - Run `uv run mypy .`
   - Run `uv run pytest` (100% pass rate).