Phase 2: Implement System Interconnections and Arithmetic Operations.

Project Standards:
- Target Python 3.10+ with strict type hints.
- Format code using PEP 8 standards with NumPy-style docstrings.
- Vectorize all polynomial and matrix calculations using NumPy (`np.polymul`, `np.polyadd`, matrix algebra). Avoid explicit Python loops.

Tasks to complete:
1. Update `TransferFunction` in `src/ctrlpy/models/transfer_function.py`:
   - Implement `__add__` and `__radd__` for parallel interconnection:
     - G1 + G2 = (num1 * den2 + num2 * den1) / (den1 * den2)
     - Support addition with scalar numbers (floats/ints).
   - Implement `__mul__` and `__rmul__` for series (cascade) interconnection:
     - G1 * G2 = (num1 * num2) / (den1 * den2)
     - Support scalar multiplication.
   - Implement `__neg__` and `__sub__`.
   - Implement `feedback(self, other=1, sign=-1)` method:
     - Closed-loop T(s) = G / (1 - sign * G * H).
     - Calculate polynomial algebra directly to avoid pole/zero blowup where possible.

2. Update `StateSpace` in `src/ctrlpy/models/state_space.py`:
   - Implement `__add__` (parallel): combine states into block diagonal [A1, 0; 0, A2], B=[B1; B2], C=[C1, C2], D=D1+D2.
   - Implement `__mul__` (series): proper state-space cascade equations.
   - Implement `feedback(self, other=None, sign=-1)` using standard state-space closed-loop formulas.

3. Create `src/ctrlpy/algebra.py`:
   - Expose public helper functions: `series(*systems)`, `parallel(*systems)`, and `feedback(sys1, sys2=1, sign=-1)`.
   - Ensure these helpers handle both `TransferFunction` and `StateSpace` seamlessly (converting or maintaining representations cleanly).
   - Export these functions in `src/ctrlpy/__init__.py`.

4. Create unit tests in `tests/test_algebra.py`:
   - Test parallel and series arithmetic of TFs against exact analytical rational functions.
   - Test feedback loop calculations (unity feedback and non-unity feedback H(s)) comparing output poles and zeros.
   - Test arithmetic operations between scalar numbers and TF/SS instances.
   - Test algebraic consistency: converting TF to SS, doing algebra in SS, converting back to TF, and asserting equality within numerical tolerance.

Verification Step:
- Run `uv run ruff check .` and `uv run ruff format --check .`
- Run `uv run pytest` to ensure all existing and new tests pass (100% pass rate).