"""ctrlpy: A Python library for classical and modern control systems analysis and design."""

__version__ = "0.1.0"

from ctrlpy.algebra import feedback, parallel, series
from ctrlpy.controllers import (
    PID,
    pd,
    pd_controller,
    pi,
    pi_controller,
    pid,
    pid_parallel,
    tune_ziegler_nichols,
)
from ctrlpy.exceptions import (
    CtrlPyError,
    DimensionMismatchError,
    UnstableSystemError,
    UnstableSystemWarning,
)
from ctrlpy.freq_domain import (
    BodeData,
    NyquistData,
    RootLocusData,
    StabilityMargins,
    bode_data,
    margin,
    nyquist_data,
    root_locus_data,
)
from ctrlpy.models.base import LTI, LinearTimeInvariant
from ctrlpy.models.state_space import StateSpace, ss
from ctrlpy.models.transfer_function import TransferFunction, tf
from ctrlpy.plotting import (
    plot_bode,
    plot_impulse,
    plot_nyquist,
    plot_root_locus,
    plot_step,
)
from ctrlpy.plotting_plotly import (
    iplot_bode,
    iplot_impulse,
    iplot_nyquist,
    iplot_root_locus,
    iplot_step,
    plot_bode_plotly,
    plot_impulse_plotly,
    plot_nyquist_plotly,
    plot_root_locus_plotly,
    plot_step_plotly,
)
from ctrlpy.simulation_results import TimeResponseData
from ctrlpy.time_domain import forced_response, impulse_response, step_response

__all__ = [
    "LTI",
    "PID",
    "BodeData",
    "CtrlPyError",
    "DimensionMismatchError",
    "LinearTimeInvariant",
    "NyquistData",
    "RootLocusData",
    "StabilityMargins",
    "StateSpace",
    "TimeResponseData",
    "TransferFunction",
    "UnstableSystemError",
    "UnstableSystemWarning",
    "__version__",
    "bode_data",
    "feedback",
    "forced_response",
    "impulse_response",
    "iplot_bode",
    "iplot_impulse",
    "iplot_nyquist",
    "iplot_root_locus",
    "iplot_step",
    "margin",
    "nyquist_data",
    "parallel",
    "pd",
    "pd_controller",
    "pi",
    "pi_controller",
    "pid",
    "pid_parallel",
    "plot_bode",
    "plot_bode_plotly",
    "plot_impulse",
    "plot_impulse_plotly",
    "plot_nyquist",
    "plot_nyquist_plotly",
    "plot_root_locus",
    "plot_root_locus_plotly",
    "plot_step",
    "plot_step_plotly",
    "root_locus_data",
    "series",
    "ss",
    "step_response",
    "tf",
    "tune_ziegler_nichols",
]
