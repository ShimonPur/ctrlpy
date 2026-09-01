Phase 5: Package Configuration, Comprehensive Documentation, and Distribution Build.

Project Standards:
- Target Python 3.10+ with full Type Annotations and clean PEP 8 formatting.
- Strict production packaging standards for distribution via uv and pip.

Tasks to complete:

1. Finalize `pyproject.toml`:
   - Set project name to "ctrlpy", version to "0.1.0", description to "A modern, high-performance Python control systems library built on NumPy and SciPy".
   - Configure build backend: use `hatchling` (or standard `flit_core`).
   - Define runtime dependencies: `dependencies = ["numpy>=1.24.0", "scipy>=1.10.0", "matplotlib>=3.7.0"]`.
   - Define development dependencies under `[project.optional-dependencies]`:
     `dev = ["pytest>=7.0.0", "ruff>=0.1.0", "ipykernel", "nbformat"]`.
   - Configure classifiers (Topics: Control Systems, Scientific/Engineering, Mathematics).

2. Finalize Public API (`src/ctrlpy/__init__.py`):
   - Expose the unified high-level interface:
     - Models: `TransferFunction`, `tf`, `StateSpace`, `ss`
     - Algebra: `series`, `parallel`, `feedback`
     - Time Analysis: `step_response`, `impulse_response`, `forced_response`, `TimeResponseData`
     - Frequency Analysis: `bode_data`, `nyquist_data`, `root_locus_data`, `margin`, `StabilityMargins`
     - Plotting: `plot_bode`, `plot_nyquist`, `plot_root_locus`, `plot_step`
   - Define explicit `__all__` list with all exported symbols.
   - Define `__version__ = "0.1.0"`.

3. Create Documentation Notebook (`notebooks/tutorial.ipynb`):
   - Create a clean Jupyter notebook using `nbformat` or direct structured JSON.
   - Build a comprehensive tutorial with Markdown theory and executable cells:
     - Section 1: Creating Models (Transfer Function & State-Space representations, LaTeX rendering).
     - Section 2: System Interconnections (Series, Parallel, Feedback loops).
     - Section 3: Time-Domain Simulations (Step & Impulse responses, extracting overshoot, rise time, settling time).
     - Section 4: Frequency-Domain Analysis (Bode plots with Gain/Phase margins, Nyquist plots with critical point, Root Locus trajectories).
     - Section 5: Model Conversions (TF <-> SS roundtrips).

4. Create `README.md`:
   - Include project badges, feature summary, and installation instructions:
     - Install locally via uv: `uv add --editable /path/to/ctrlpy`
     - Install via pip: `pip install .`
   - Provide a complete copy-paste Quickstart script covering model creation, closed-loop feedback, step response metrics, and plotting Bode with margins.
   - Detail comparison vs legacy python-control (vectorized performance, modern dataclasses, clean type hints).

5. Verification and Packaging Build:
   - Run `uv run ruff format .` and `uv run ruff check .`
   - Run `uv run pytest` to ensure all tests pass (100% pass rate).
   - Run `uv build` to compile the final `.whl` and `.tar.gz` packages into the `dist/` directory.

Execute these tasks cleanly.