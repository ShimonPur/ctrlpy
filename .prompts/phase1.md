Phase 1: Implement Core LTI Models and Representations.

Project Standards:
- Target Python 3.10+ with strict type hints (use `typing.Sequence`, `numpy.typing.NDArray`).
- Format code using PEP 8 standards with NumPy-style docstrings.
- Core math should rely on NumPy vectorization and SciPy (`scipy.signal.tf2ss`, `scipy.signal.ss2tf`).

Tasks to complete:
1. Create `src/ctrlpy/models/base.py`:
   - Define an abstract base class `LinearTimeInvariant` (or `LTI`) inheriting from `abc.ABC`.
   - Define common properties/methods: `inputs`, `outputs`, `is_siso`, `poles()`, `zeros()`.

2. Create `src/ctrlpy/models/transfer_function.py`:
   - Implement class `TransferFunction(LinearTimeInvariant)` with attributes `num` (numerator) and `den` (denominator) stored as normalized 1D NumPy arrays (float64).
   - Support initialization from lists/arrays, e.g., `TransferFunction([1], [1, 2, 1])`.
   - Add a method `to_ss()` that returns a `StateSpace` instance using `scipy.signal.tf2ss`.
   - Implement `__repr__`, `__str__`, and `_repr_latex_` for clean mathematical representation in Jupyter environments (formatting standard rational functions in 's').

3. Create `src/ctrlpy/models/state_space.py`:
   - Implement class `StateSpace(LinearTimeInvariant)` with attributes `A`, `B`, `C`, `D` as 2D NumPy arrays (float64).
   - Validate matrix dimensions upon initialization (check compatibility of A, B, C, D).
   - Add a method `to_tf()` that returns a `TransferFunction` instance using `scipy.signal.ss2tf`.
   - Implement `__repr__`, `__str__`, and `_repr_latex_`.

4. Update `src/ctrlpy/models/__init__.py` and `src/ctrlpy/__init__.py`:
   - Export `TransferFunction` (with alias `tf`) and `StateSpace` (with alias `ss`).

5. Create comprehensive test suites in `tests/test_models.py`:
   - Test proper instantiation and dimension validation (including invalid inputs).
   - Test `poles()` and `zeros()` calculation against analytical values.
   - Test round-trip conversion: `tf.to_ss().to_tf()` matching original coefficients within tolerance (`pytest.approx`).
   - Test LaTeX string formatting output.

Verification Step:
- Run `uv run ruff check .` and `uv run ruff format --check .` to ensure zero lint/style errors.
- Run `uv run pytest` to ensure 100% test pass rate.