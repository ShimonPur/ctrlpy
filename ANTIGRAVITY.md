# Engineering Standards & Architecture: `ctrlpy`

You are building `ctrlpy`, a high-performance, strictly typed Python control systems library. Follow these mandates across all phases:

## 1. Architectural Principles
- Layout: Modern `src/ctrlpy/` structure.
- Numerical Core: NumPy, SciPy, Matplotlib, Plotly.
- Symbolic & Pedagogical Engine: Strictly isolated in `src/ctrlpy/symbolic/`. Protect imports with safe exception guards if SymPy is missing.
- Jupyter Integration: Implement `_repr_latex_()` and `_repr_markdown_()` on all mathematical and pedagogical objects.
- Fluent/OOP API: Direct convenience methods on systems (e.g., `.step()`, `.bode()`, `.nyquist()`, `.rlocus()`).

## 2. Mandatory Documentation & Notebook Synchronization Policy
Every phase that adds or modifies a feature MUST synchronously update:
1. Docstrings: Full NumPy format with embedded LaTeX ($...$ / $$...$$).
2. MkDocs Documentation Site: Add or update relevant pages under `docs/` and verify `mkdocs build --strict` passes.
3. Jupyter Notebooks: Add or update end-to-end examples under `notebooks/` and verify they execute without errors.

## 3. Tooling & Quality Gates
Always run and pass with zero warnings/errors:
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src/ctrlpy`
- `uv run pytest --cov=src/ctrlpy --cov-fail-under=90`
- `uv run mkdocs build --strict`