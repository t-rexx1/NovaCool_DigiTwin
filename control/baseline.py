"""
Baseline controller: fixed CRAH setpoints for the full 24 hrs window.

This is the no control reference case. So all CRAHs run at:-
    - Supply temp.: 7 deg C (chilled water design point)
    - Fan speed: 100% (means full airflow)

Used to establish a performance baseline that the heuristic controller is compared against (Mainly task 4).
"""

import numpy as np


class FixedSetpointController:
    """
    Baseline controller with constant CRAH setpoints.

    No feedback ==> operates in open loop regardless of thermal state.

    This represents the doing nothing case that autonomous controllers must essentially outperform!
    """

    def __init__(
        self,
        supply_temp: float = 7.0,    # deg C
        fan_frac: float = 1.0,       # full speed
        n_crahs: int = 4,
    ):
        self.supply_temp = supply_temp
        self.fan_frac = fan_frac
        self.n_crahs = n_crahs

    def get_setpoints(self, t: int, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Return fixed setpoints regardless of current state.

        Parameters
        ----------
        t   : Current timestep                  (ignored - open loop)
        obs : Current thermal/power observation (ignored - open loop)

        Returns
        -------
        crah_supply_temps : (n_crahs,) degC -   all identical
        crah_fan_fracs    : (n_crahs,) [0-1] -  all identical
        """
        return (np.full(self.n_crahs, self.supply_temp),
                np.full(self.n_crahs, self.fan_frac),)

    def run_episode(self, facility) -> np.ndarray:
        """
        Generate full (1440, 4) control arrays for facility.run().

        Returns
        -------
        supply_temps : (1440, 4)
        fan_fracs    : (1440, 4)
        """
        supply = np.full((facility.N_STEPS, self.n_crahs), self.supply_temp)
        fans   = np.full((facility.N_STEPS, self.n_crahs), self.fan_frac)
        return supply, fans