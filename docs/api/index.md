# API Reference Overview

Welcome to the `ctrlpy` API reference documentation.

`ctrlpy` is organized into focused, modular subpackages:

| Module | Description |
| :--- | :--- |
| [`ctrlpy.models`](models.md) | Linear Time-Invariant (LTI) base classes, `TransferFunction`, and `StateSpace`. |
| [`ctrlpy.algebra`](algebra.md) | Block diagram algebra (`series`, `parallel`, `feedback`). |
| [`ctrlpy.time_domain`](time_domain.md) | Step, impulse, and forced response solvers and `TimeResponseData`. |
| [`ctrlpy.freq_domain`](freq_domain.md) | Frequency responses (`bode_data`, `nyquist_data`, `root_locus_data`, `margin`). |
| [`ctrlpy.controllers`](controllers.md) | PID, PI, PD controller constructors and Ziegler-Nichols tuning. |
| [`ctrlpy.plotting`](plotting.md) | Matplotlib static plots and Plotly interactive visualizations. |
| [`ctrlpy.pedagogy`](pedagogy.md) | Routh-Hurwitz table, analytical Root Locus rules, and steady-state error analysis. |
| [`ctrlpy.exceptions`](exceptions.md) | Custom exception and warning hierarchy. |

