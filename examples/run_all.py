"""Regenerate every figure and table in ``outputs/`` (takes 1-2 hours with the default settings)."""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
steps = [
    ["01_gp_regression_illustration.py"],
    ["02_simulation_study.py", "--reps", "10"],
    ["03_us_uncertainty_gpvar8.py"],
    ["04_forecast_evaluation.py"],
]
for s in steps:
    print("=" * 78, "\n>>>", " ".join(s), flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, s[0])] + s[1:], check=True)
