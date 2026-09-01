# Engineering Skill & Mandates: spectro Library

You are the autonomous engineering agent building spectro on Windows using uv.

## Architecture & Layout

- Layout: Modern src/spectro/ layout (core, dsp, csp, viz, symbolic).
- Pure Numerical Core: core, dsp, csp depend strictly on NumPy, SciPy, Matplotlib, and Plotly.
- Isolated Symbolic Module: src/spectro/symbolic/ wraps SymPy. Imports of SymPy must be lazy or safely guarded with a clear ImportError suggesting: uv add spectro --extra symbolic.
- Dual-backend visualization: .plot*\*() returns Matplotlib (fig, ax), .iplot*\*() returns Plotly go.Figure.
- Jupyter notebook support: implement _repr_latex_() and _repr_markdown_() on mathematical objects.

## Tooling, Typing & Execution Environment

- Runner: Always run checks with uv run (e.g., uv run pytest, uv run ruff check ., uv run mypy src/spectro).
- Static Typing: PEP 484 type annotations on every public and private interface. Must pass mypy --strict.
- Linting: Line length 100, managed via Ruff (ruff.toml).
- Documentation: NumPy docstrings with LaTeX formulas.
- Testing: Comprehensive Pytest suite under tests/ targeting >90% coverage.
