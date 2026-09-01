Phase 7: Packaging Infrastructure, CI/CD Setup, Configuration Audits, and Practical Engineering Case Studies.

Project Standards:
- Target Python 3.10+ with full type annotations.
- Modern packaging standards (compliant with pyproject.toml, hatchling backend, uv workflows).
- Real-world engineering case studies with rich physical explanations.

Tasks to complete:

1. Audit and Finalize Packaging & Tool Configuration:
   - `pyproject.toml`:
     - Validate all project metadata (name="ctrlpy", version="0.1.0", authors, license, readme="README.md").
     - Define full runtime dependencies (`numpy>=1.24.0`, `scipy>=1.10.0`, `matplotlib>=3.7.0`, `plotly>=5.15.0`).
     - Define optional/dev dependencies under `[project.optional-dependencies]`:
       `dev = ["pytest>=7.0.0", "pytest-cov>=4.1.0", "ruff>=0.1.0", "mypy>=1.5.0", "ipykernel", "nbformat"]`.
     - Configure tool sections:
       - `[tool.ruff]`: line-length = 100, target-version = "py310".
       - `[tool.mypy]`: strict = true, ignore_missing_imports = true for third-party libraries without stubs.
       - `[tool.pytest.ini_options]`: addopts = "--cov=src/ctrlpy --cov-report=term-missing".
   - Create `.gitignore`:
     - Standard Python/uv ignore rules (`__pycache__/`, `.venv/`, `dist/`, `.pytest_cache/`, `.mypy_cache/`, `.coverage`, `.ipynb_checkpoints/`).

2. Setup CI/CD Infrastructure (`.github/workflows/ci.yml`):
   - GitHub Actions workflow running on `push` and `pull_request` to `main`/`master`.
   - Test matrix across Python 3.10, 3.11, 3.12 on `ubuntu-latest`.
   - Workflow steps:
     1. Checkout code.
     2. Install `uv`.
     3. Install dependencies: `uv sync --all-extras`.
     4. Lint check: `uv run ruff check .` and `uv run ruff format --check .`.
     5. Type check: `uv run mypy .`.
     6. Run test suite with coverage: `uv run pytest`.
     7. Test package distribution build: `uv build`.

3. Reorganize Notebooks into Practical Engineering Case Studies:
   - Remove/replace old monolithic notebooks with a clean, structured directory `notebooks/`:
     - `01_quickstart.ipynb`: Basic model creation, interconnections (series, parallel, feedback), and fluent plotting.
     - `02_dc_motor_control.ipynb`: Practical DC Motor modeling ($J, b, K_t, R, L$), converting physical parameters to Transfer Function, step response, and PI controller design for speed tracking.
     - `03_mass_spring_damper.ipynb`: Mechanical 2nd-order oscillator, Root Locus parameter sweeps with Plotly, and PID tuning to achieve specific damping ratio $\zeta$ and settling time.
     - `04_frequency_domain_deep_dive.ipynb`: Deep dive into Bode margins ($W_{cg}, W_{cp}$) and Nyquist stability criteria with integrator contour visualization using Plotly interactive charts.
   - Ensure all notebooks execute cleanly and generate outputs without errors.

4. Update `README.md`:
   - Add status badges (CI, Python versions, License, Code style: Ruff).
   - Document project structure, installation instructions, and provide a clear index linking to each practical notebook case study.

5. Full Verification:
   - Run `uv run ruff format .` and `uv run ruff check .`
   - Run `uv run mypy .`
   - Run `uv run pytest` to ensure >90% test coverage and 100% test pass rate.
   - Run `uv build` to confirm wheel/sdist packages build cleanly.