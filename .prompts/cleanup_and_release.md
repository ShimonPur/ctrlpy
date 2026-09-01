# Phase: Project Purge, Documentation Polish & Terminal/Notebook Release

You are tasked with completely stripping all GUI and diagram-engine artifacts from the repository, validating notebook execution, and preparing a clean, stable terminal/Jupyter-first release. Follow `ANTIGRAVITY.md` strictly.

## Action Items:

1. **Complete Removal of GUI & Diagram Engine Artifacts:**
   - Delete all GUI and visual modules: remove `src/ctrlpy/gui/` and `src/ctrlpy/diagram/` (or any GUI/diagram source directories).
   - Delete all corresponding tests: remove `tests/test_gui.py`, `tests/test_diagram.py`, and any visual routing test files.
   - Clean `pyproject.toml`:
     * Remove `PySide6`, `shiboken6`, or any GUI-specific optional dependencies/entry points (`[project.scripts]` like `ctrlpy-gui`).
     * Ensure only core mathematical/scientific dependencies (NumPy, SciPy, Matplotlib, Plotly, optional SymPy) and development tools remain.
   - Run `uv sync --all-extras` to prune orphaned dependencies.

2. **Documentation & README Overhaul:**
   - Update `README.md` to reflect a pure CLI / Jupyter scientific workflow.
   - Remove all screenshots, diagrams, and mentions of GUI, canvas, Manhattan routing, or desktop apps.
   - Add clear Quickstart examples showcasing the mathematical API in Python scripts and Jupyter notebooks.
   - Ensure all public functions and mathematical objects have complete NumPy-style docstrings with embedded LaTeX formulas ($...$ / $$...$$).

3. **Notebook & Example Validation:**
   - Verify that all example notebooks in `notebooks/` (or docstring code snippets) run without errors.
   - Remove any deprecated imports referencing `ctrlpy.gui` or `ctrlpy.diagram`.
   - Ensure notebooks execute cleanly with valid LaTeX equations, Matplotlib/Plotly figures, and rich markdown rendering (`_repr_latex_`, `_repr_markdown_`).

4. **Quality Gates & Verification:**
   Run and fix every issue until all of the following commands pass with zero warnings and zero errors:
   - `uv run ruff check .`
   - `uv run ruff format --check .`
   - `uv run mypy src/`
   - `uv run pytest --cov=src/ --cov-fail-under=90`