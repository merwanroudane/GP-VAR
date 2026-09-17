"""
gpvar - Gaussian Process Vector Autoregressions in Python
=========================================================

A faithful, self-contained implementation of the GP-VAR with stochastic
volatility of Hauzenberger, Huber, Marcellino and Petz (2022),
"Gaussian Process Vector Autoregressions and Macroeconomic Uncertainty"
(arXiv:2112.01995), including the discrete hyperparameter grid, the
conjugate SV-scaled kernels, the independence-MH volatility sampler,
generalized impulse responses with sign/size/time asymmetries, forecasting,
density-forecast evaluation and journal-style figures and tables.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
import warnings as _warnings

# SciPy emits a (harmless) version warning with very recent NumPy releases
_warnings.filterwarnings("ignore", message="A NumPy version >=.*is required for this version of SciPy")

from .model import GPVAR, GPVARResults
from .girf import girf, girf_sign_asymmetry, girf_size_asymmetry, GIRFResult, ordering_robustness
from .forecast import predict, log_predictive_score, crps, recursive_forecast_evaluation, lpbf_table, ForecastResult
from .simulate import simulate_paper_dgp, simulate_replication_dgp, SimulatedData
from .gp import GPRegression
from .priors import SVPriors, HorseshoeState
from .benchmarks import BVAR
from . import data, plots, tables, kernels, sv

__version__ = "1.0.0"
__author__ = "Merwan Roudane"
__all__ = [
    "GPVAR", "GPVARResults", "girf", "girf_sign_asymmetry", "girf_size_asymmetry", "GIRFResult",
    "ordering_robustness", "predict", "log_predictive_score", "crps", "recursive_forecast_evaluation",
    "lpbf_table", "ForecastResult", "simulate_paper_dgp", "simulate_replication_dgp", "SimulatedData",
    "GPRegression", "SVPriors", "HorseshoeState", "BVAR", "data", "plots", "tables", "kernels", "sv",
]
