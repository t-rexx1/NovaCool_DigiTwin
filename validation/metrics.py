"""
This code details model validation and calibration for NovaCool digital twin.

Compares simulation outputs against sensor_reference.csv.
Again this follows Prof Zohdi's (2022) validation methodology:
    A digital twin must be validated against real (or synthetic reference) data before it can be trusted as a controller or design tool.

Metrics:
    RMSE  - penalises large errors heavily (squared)
    MAE   - average absolute error (interpretable in degC)
    MaxAE - worst case error (critical for thermal safety analysis)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union


def load_sensor_reference(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load sensor_reference.csv.

    Columns: timestamp, rack_id, inlet_temp_c, outlet_temp_c, pdu_power_kw
    Returns DataFrame with 288,000 rows (200 racks x 1440 timesteps).
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["timestamp", "rack_id"]).reset_index(drop=True)
    return df


def sensor_to_arrays(sensor_df: pd.DataFrame) -> dict:
    """
    Pivoting sensor DataFrame into (1440, 200) arrays for vector comparison.

    Returns
    -------
    dict with keys:
        inlet_temp_c  : (1440, 200)
        outlet_temp_c : (1440, 200)
        pdu_power_kw  : (1440, 200)
    """
    def _pivot(col):
        return (
            sensor_df
            .pivot(index="timestamp", columns="rack_id", values=col)
            .sort_index()
            .sort_index(axis=1)
            .values.astype(np.float64)
        )

    return {
        "inlet_temp_c":  _pivot("inlet_temp_c"),
        "outlet_temp_c": _pivot("outlet_temp_c"),
        "pdu_power_kw":  _pivot("pdu_power_kw"),
    }


def rmse(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Root Mean Squared Error [same units as input]."""
    return float(np.sqrt(np.mean((predicted - reference) ** 2)))


def mae(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Mean Absolute Error [same units as input]."""
    return float(np.mean(np.abs(predicted - reference)))


def max_absolute_error(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Maximum Absolute Error — worst-case deviation [same units as input]."""
    return float(np.max(np.abs(predicted - reference)))


def compute_all_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    label: str = "outlet_temp_c",
) -> dict:
    """
    Computing RMSE, MAE, MaxAE for a predicted vs reference array pair.

    Parameters-
    predicted  : Simulation output array, any shape
    reference  : Reference sensor array, same shape as predicted
    label      : Name of the variable being compared

    Returns-
    dict with keys: label, rmse, mae, max_absolute_error, n_samples
    """
    assert predicted.shape == reference.shape, (
        f"Shape mismatch: predicted {predicted.shape} vs "
        f"reference {reference.shape}"
    )
    return {
        "label":             label,
        "rmse":              rmse(predicted, reference),
        "mae":               mae(predicted, reference),
        "max_absolute_error": max_absolute_error(predicted, reference),
        "n_samples":         predicted.size,
    }


def error_by_timestep(
    predicted: np.ndarray,   # (1440, 200)
    reference: np.ndarray,   # (1440, 200)
) -> np.ndarray:
    """
    Compute MAE at each timestep across all racks.
    Returns shape (1440,) — useful for identifying divergence windows.
    """
    return np.mean(np.abs(predicted - reference), axis=1)


def find_divergence_windows(
    timestep_errors: np.ndarray,   # (1440,)
    threshold_percentile: float = 90.0,
    window_minutes: int = 30,
) -> list[dict]:
    """
    Identify contiguous time windows where model error is highest.
    Used for root-cause analysis in write-up (Task 3).

    Parameters-
    timestep_errors      : MAE per timestep (1440,)
    threshold_percentile : Flag timesteps above this error percentile
    window_minutes       : Minimum window size to report

    Returns-
    List of dicts: {start_min, end_min, mean_error, max_error}
    """
    threshold = np.percentile(timestep_errors, threshold_percentile)
    high_error = timestep_errors > threshold

    windows = []
    in_window = False
    start = 0

    for t in range(len(high_error)):
        if high_error[t] and not in_window:
            in_window = True
            start = t
        elif not high_error[t] and in_window:
            in_window = False
            if (t - start) >= window_minutes:
                windows.append({
                    "start_min": start,
                    "end_min":   t,
                    "start_hhmm": f"{start // 60:02d}:{start % 60:02d}",
                    "end_hhmm":   f"{t // 60:02d}:{t % 60:02d}",
                    "mean_error": float(timestep_errors[start:t].mean()),
                    "max_error":  float(timestep_errors[start:t].max()),
                })

    return windows


def run_validation(
    facility,
    sensor_csv: Union[str, Path],
) -> dict:
    """
    Full validation pipeline: run baseline simulation, compare to reference.

    Parameters-
    facility   : NovaCoolFacility instance (already loaded with workload)
    sensor_csv : Path to sensor_reference.csv

    Returns-
    dict with keys:
        metrics_outlet  : RMSE/MAE/MaxAE for outlet temperature
        metrics_inlet   : RMSE/MAE/MaxAE for inlet temperature
        timestep_errors : (1440,) MAE per timestep
        divergence_windows : list of high-error windows
        results_df      : Full simulation results DataFrame
        sensor_arrays   : Pivoted sensor reference arrays
    """
    # Run baseline simulation (fixed setpoints)
    results_df = facility.run()

    # Load and reshape sensor reference
    sensor_df = load_sensor_reference(sensor_csv)
    sensor_arrays = sensor_to_arrays(sensor_df)

    # Building sim outlet/inlet arrays (1440, 200) from per timestep data
    # We need to re run and capture per rack arrays
    sim_outlet = _collect_per_rack(facility, "outlet_temp_c")
    sim_inlet  = _collect_per_rack(facility, "inlet_temp_c")

    # Compute the metrics...
    metrics_outlet = compute_all_metrics(
        sim_outlet, sensor_arrays["outlet_temp_c"], label="outlet_temp_c"
    )
    metrics_inlet = compute_all_metrics(
        sim_inlet,  sensor_arrays["inlet_temp_c"],  label="inlet_temp_c"
    )

    # Per timestep error for doing divergence analysis
    timestep_errors = error_by_timestep(
        sim_outlet, sensor_arrays["outlet_temp_c"]
    )

    divergence_windows = find_divergence_windows(timestep_errors)

    return {
        "metrics_outlet":      metrics_outlet,
        "metrics_inlet":       metrics_inlet,
        "timestep_errors":     timestep_errors,
        "divergence_windows":  divergence_windows,
        "results_df":          results_df,
        "sensor_arrays":       sensor_arrays,
        "sim_outlet":          sim_outlet,
        "sim_inlet":           sim_inlet,
    }


def _collect_per_rack(facility, field: str) -> np.ndarray:
    """
    Re-run simulation and collect per-rack temperature arrays.
    Returns shape (1440, 200).
    """
    import numpy as np
    from simulator.thermal import ThermalConfig

    cfg = facility.thermal.cfg
    shape = (facility.N_STEPS, cfg.n_crahs)
    T_supply = np.full(shape, 7.0)
    fan_f    = np.ones(shape)

    out = np.zeros((facility.N_STEPS, cfg.n_racks))
    for t in range(facility.N_STEPS):
        rack_power = facility.workload[t]
        th = facility.thermal.step(rack_power, T_supply[t], fan_f[t])
        out[t] = th[field]
    return out


if __name__ == "__main__":
    from simulator.facility import NovaCoolFacility

    facility = NovaCoolFacility.from_csvs("data/workload_trace.csv")
    val = run_validation(facility, "data/sensor_reference.csv")

    print("\n=== Outlet Temperature Validation ===")
    m = val["metrics_outlet"]
    print(f"  RMSE : {m['rmse']:.4f} degC")
    print(f"  MAE  : {m['mae']:.4f} degC")
    print(f"  MaxAE: {m['max_absolute_error']:.4f} degC")

    print("\n=== Inlet Temperature Validation ===")
    m = val["metrics_inlet"]
    print(f"  RMSE : {m['rmse']:.4f} degC")
    print(f"  MAE  : {m['mae']:.4f} degC")
    print(f"  MaxAE: {m['max_absolute_error']:.4f} degC")

    print("\n=== Divergence Windows (top error periods) ===")
    for w in val["divergence_windows"]:
        print(f"  {w['start_hhmm']} - {w['end_hhmm']} | "
              f"mean={w['mean_error']:.3f} degC | "
              f"max={w['max_error']:.3f} degC")