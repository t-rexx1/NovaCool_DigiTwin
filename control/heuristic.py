"""
Heuristic thermostat controller for NovaCool.

Design philosophy (Zohdi 2022 adaptive control):
    The controller observes the system state at each timestep and adjusts the CRAH supply temperature to keep the maximum rack outlet temp within a safe dead band.

Dead band logic used:
    if max_outlet > T_high:  decrease supply temp (then cool more aggressively)
    if max_outlet < T_low:   increase supply temp (save energy)
    else:                    hold current setpoint

This is analogous to Prof. Zohdi's adaptive digital framework where the supplier adjusts output based on device state feedback.

Fan speed is also modulated such:
    High thermal load --> ramp fans up (more airflow = more cooling)
    Low thermal load  --> ramp fans down (cube law --> big energy savings)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HeuristicConfig:
    # Dead band thresholds [degC]
    T_high: float = 37.0       # above this: cool more
    T_low: float = 33.0        # below this: relax cooling

    # Supply temperature adjustment per timestep [deg C]
    temp_step: float = 0.5

    # Supply temperature bounds [degC]
    T_supply_min: float = 6.0
    T_supply_max: float = 18.0

    # Fan speed adjustment per timestep [fraction]
    fan_step: float = 0.05

    # Fan speed bounds [fraction]
    fan_min: float = 0.3       # never to go below 30% (airflow floor limit)
    fan_max: float = 1.0

    n_crahs: int = 4


class ThermostatController:
    """
    Reactive dead band thermostat controller...

    Observes max rack outlet temperature and adjusts CRAH supply temperature and fan speed to maintain thermal safety while minimising cooling energy (so PUE improvement).

    Each CRAH is controlled independently based on the max outlet temperature of its assigned racks.
    """

    def __init__(self, config: Optional[HeuristicConfig] = None):
        self.cfg = config or HeuristicConfig()
        # State: current setpoints (basically updated each timestep)
        self.T_supply = np.full(self.cfg.n_crahs, 7.0)
        self.fan_fracs = np.ones(self.cfg.n_crahs)

    def reset(self):
        """Reset controller state to initial conditions."""
        self.T_supply  = np.full(self.cfg.n_crahs, 7.0)
        self.fan_fracs = np.ones(self.cfg.n_crahs)

    def act(
        self,
        max_outlet_per_crah: np.ndarray,  # (n_crahs,) max outlet per zone
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute new CRAH setpoints based on observed outlet temperatures.

        Parameters
        ----------
        max_outlet_per_crah : Max rack outlet temp per CRAH zone [degC]

        Returns
        -------
        T_supply  : Updated supply temp per CRAH [degC]
        fan_fracs : Updated fan speed per CRAH [0-1]
        """
        for j in range(self.cfg.n_crahs):
            T_max = max_outlet_per_crah[j]

            # Supply temp adjustment
            if T_max > self.cfg.T_high:
                # Too hot!: decrease supply temp (more cooling)
                self.T_supply[j] -= self.cfg.temp_step
            elif T_max < self.cfg.T_low:
                # Well within limit: increase supply temp (save energy)
                self.T_supply[j] += self.cfg.temp_step
            # else: within dead band, holdthe current setpoint

            # Fan speed adjustment 
            if T_max > self.cfg.T_high:
                # Also ramp up the fans for faster response
                self.fan_fracs[j] += self.cfg.fan_step
            elif T_max < self.cfg.T_low:
                # Ramp down fans to exploit cube law energy savings
                self.fan_fracs[j] -= self.cfg.fan_step

        # Clip to physical bounds
        self.T_supply  = np.clip(self.T_supply,  self.cfg.T_supply_min, self.cfg.T_supply_max)
        self.fan_fracs = np.clip(self.fan_fracs,  self.cfg.fan_min,      self.cfg.fan_max)

        return self.T_supply.copy(), self.fan_fracs.copy()

    def run_episode(self, facility) -> tuple[np.ndarray, np.ndarray]:
        """
        Run full 24 hr episode with closed-loop control.

        At each timestep:
            - Run thermal model with current setpoints
            - Observe max outlet temp per CRAH zone
            - Update setpoints via dead-band logic
            - Store setpoints for facility.run() replay

        Returns
        -------
        supply_temps : (1440, 4) — time-varying supply temp schedule
        fan_fracs    : (1440, 4) — time-varying fan speed schedule
        """
        self.reset()

        supply_schedule = np.zeros((facility.N_STEPS, self.cfg.n_crahs))
        fan_schedule    = np.zeros((facility.N_STEPS, self.cfg.n_crahs))

        # Ini. setpoints
        T_supply  = self.T_supply.copy()
        fan_fracs = self.fan_fracs.copy()

        for t in range(facility.N_STEPS):
            # Store current setpoints
            supply_schedule[t] = T_supply
            fan_schedule[t]    = fan_fracs

            # Step the thermal model to observe state
            rack_power = facility.workload[t]
            th = facility.thermal.step(rack_power, T_supply, fan_fracs)

            # Compute max outlet temp per CRAH zone
            crah_idx = facility.thermal._crah_idx
            max_outlet_per_crah = np.array([
                th["outlet_temp_c"][crah_idx == j].max()
                for j in range(self.cfg.n_crahs)
            ])

            # Update setpoints for the next timestep
            T_supply, fan_fracs = self.act(max_outlet_per_crah)

        return supply_schedule, fan_schedule