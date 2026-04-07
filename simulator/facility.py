"""
NovaCool facility orchestrator.

This code bascially Couples the thermal and power models and runs the full 24-hour simulation.

Following Zohdi (2022): the facility is the supplier + coupled device system where each rack is a device and the CRAHs are the supplier subsystem.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from simulator.thermal import ThermalModel, ThermalConfig
from simulator.power import PowerModel, PowerConfig


def load_workload(csv_path: Union[str, Path]) -> np.ndarray:
    """
    Load workload_trace.csv and return a (1440, 200) array.
    Rows = timesteps (minutes), Columns = rack IDs.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    pivoted = df.pivot(
        index="timestamp", columns="rack_id", values="power_kw"
    )
    pivoted = pivoted.sort_index().sort_index(axis=1)
    assert pivoted.shape[0] == 1440, (
        f"Expected 1440 timesteps, got {pivoted.shape[0]}"
    )
    return pivoted.values.astype(np.float64)  # (1440, n_racks)


def load_sensor_reference(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load sensor_reference.csv for model validation."""
    return pd.read_csv(csv_path, parse_dates=["timestamp"])


class NovaCoolFacility:
    """
    Full NovaCool 10 MW data center digital twin.

    Orchestrates thermal + power models at each 1-minute timestep.
    Control inputs (CRAH supply temps, fan speeds) can be:
      - Fixed scalars (baseline: constant setpoints)
      - 1D arrays shape (n_crahs,)  (time invariant policy)
      - 2D arrays shape (1440, n_crahs)  (and time varying policy)

    Usage
    -----
        facility = NovaCoolFacility.from_csvs("data/workload_trace.csv")
        results = facility.run()          # baseline 24h run
        results.head()
    """

    N_STEPS = 1440  #24 hrs x 60 mins

    def __init__(
        self,
        workload: np.ndarray,              # (1440, 200)
        thermal_config: Optional[ThermalConfig] = None,
        power_config: Optional[PowerConfig] = None,
    ):
        assert workload.shape[0] == self.N_STEPS, (
            f"Workload must have 1440 timesteps, got {workload.shape[0]}"
        )
        self.workload = workload
        n_racks = workload.shape[1]
        if thermal_config is None:
            thermal_config = ThermalConfig(n_racks=n_racks)
        elif thermal_config.n_racks != n_racks:
            thermal_config.n_racks = n_racks
        self.thermal = ThermalModel(thermal_config)
        self.power = PowerModel(power_config)

    @classmethod
    def from_csvs(
        cls,
        workload_csv: Union[str, Path],
        thermal_config: Optional[ThermalConfig] = None,
        power_config: Optional[PowerConfig] = None,
    ) -> "NovaCoolFacility":
        """Convenience constructor from CSV path."""
        workload = load_workload(workload_csv)
        return cls(workload, thermal_config, power_config)

    def _resolve_control(
        self,
        arr: Optional[np.ndarray],
        default_val: float,
        shape: tuple,
    ) -> np.ndarray:
        """
        Normalize control input to shape (1440, n_crahs).
        Accepts: None, scalar, 1D (n_crahs,), or 2D (1440, n_crahs).
        """
        n_crahs = shape[1]
        if arr is None:
            return np.full((self.N_STEPS, n_crahs), default_val)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(shape, float(arr))
        if arr.ndim == 1:
            assert len(arr) == n_crahs
            return np.tile(arr, (self.N_STEPS, 1))
        assert arr.shape == shape, f"Expected {shape}, got {arr.shape}"
        return arr

    def run(
        self,
        crah_supply_temps: Optional[np.ndarray] = None,
        crah_fan_fracs: Optional[np.ndarray] = None,
        sensor_noise_std: float = 0.0,   #EDIT:  Adds sensor noise for stochasticity to simulation as part of stretch goal...
    ) -> pd.DataFrame:
        """
        Runs the full 24 hr simulation.

        Parameters
        ----------
        crah_supply_temps : CRAH supply temperature setpoints [degC]
                            Default: 7.0 degC for all CRAHs all timesteps
        crah_fan_fracs    : CRAH fan speed fractions [0-1]
                            Default: 1.0 (full speed) for all CRAHs

        Returns
        -------
        pd.DataFrame with 1440 rows and columns:
            t, avg_cold_aisle_c, avg_hot_aisle_c, peak_outlet_c,
            total_cooling_kw, total_it_kw, pue, total_facility_kw,
            n_violations, ups_efficiency, fan_power_kw, pump_power_kw
        """
        shape = (self.N_STEPS, self.thermal.cfg.n_crahs)
        T_supply = self._resolve_control(crah_supply_temps, 7.0, shape)
        fan_f    = self._resolve_control(crah_fan_fracs, 1.0, shape)

        records = []
        for t in range(self.N_STEPS):
            rack_power = self.workload[t]   # (200,)
            th = self.thermal.step(rack_power, T_supply[t], fan_f[t])
            # EDit: Add sensor noise (stochastic simulation)
            if sensor_noise_std > 0:
                noise = np.random.normal(0, sensor_noise_std, th["outlet_temp_c"].shape)
                th["outlet_temp_c"] = th["outlet_temp_c"] + noise
            pw = self.power.step(rack_power.sum(), th["crah_cooling_kw"], fan_f[t])

            records.append({
                "t":                 t,
                "avg_cold_aisle_c":  th["inlet_temp_c"].mean(),
                "avg_hot_aisle_c":   th["hot_aisle_temp_c"].mean(),
                "peak_outlet_c":     th["outlet_temp_c"].max(),
                "total_cooling_kw":  th["crah_cooling_kw"].sum(),
                "total_it_kw":       rack_power.sum(),
                "pue":               pw["pue"],
                "total_facility_kw": pw["total_facility_kw"],
                "n_violations":      self.thermal.n_violations(th["outlet_temp_c"]),
                "ups_efficiency":    pw["ups_efficiency"],
                "fan_power_kw":      pw["fan_power_kw"],
                "pump_power_kw":     pw["pump_power_kw"],
            })

        return pd.DataFrame(records)


if __name__ == "__main__":
    # running a quick smoke test... run from repo root: python -m simulator.facility
    import matplotlib.pyplot as plt

    facility = NovaCoolFacility.from_csvs("data/workload_trace.csv")
    results = facility.run()

    print(results[["avg_cold_aisle_c", "avg_hot_aisle_c",
                   "peak_outlet_c", "pue"]].describe())
    print(f"\nTotal violations: {results['n_violations'].sum()}")
    print(f"Mean PUE: {results['pue'].mean():.3f}")
    print(f"Peak outlet max: {results['peak_outlet_c'].max():.2f} degC")