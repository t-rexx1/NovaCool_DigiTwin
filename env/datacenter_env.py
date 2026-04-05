"""
DataCenterEnv: Gym style RL training environment for NovaCool.

Design philosophy:
    In this environment we wrap the digital twin model and expose a clean step/reset API so that RL engineers/team can train control agents without needing to know the simulation internals.

    Following Zohdi methodology again (from 2022 paper): the digital twin must run faster than real time to be viable as a training environment!! 
    
    So Our 1440 steps episode completes in <1 second which is well within that requirement :)

Observation space (13 dim):
    [0]   avg_cold_aisle_temp_c     (deg C, normalised/20)
    [1]   avg_hot_aisle_temp_c      (deg C, normalised/50)
    [2]   max_rack_outlet_temp_c    (deg C, normalised/50)
    [3]   total_it_load_kw          (kW,   normalised/10000)
    [4]   time_of_day               (minutes, normalised/1440)
    [5:9] crah_supply_temps x4      (deg C, normalised/20)
    [9:13] crah_fan_fracs x4        (fraction, already b/w 0-1)

Action space (8 dim):
    [0:4] delta_supply_temp x4      (deg C/step, clipped +-2)
    [4:8] delta_fan_frac x4         (frac/step, clipped +- 0.1)

Reward function (I hv explained this in write up):
    r = w_throughput * IT_load
      - w_violation * max(0, max_outlet - 40)^2
      - w_energy * total_facility_kw

    Violation term is SQUARED (exponential) to create strong non linear penalty near the 40 deg Celsius hard constraint --> so the agent will learn to stay well clear of this limit.
    Throughput and energy terms create a soft Pareto front.
"""

import numpy as np
from typing import Optional, Tuple


class DataCenterEnv:
    """
    Gym-style env wrapping the NovaCool digital twin.

    Follows standard Gym interface:
        obs = env.reset()
        obs, reward, done, info = env.step(action)
    [This was referenced in the case study pdf]

    Does NOT inherit from gym.Env to avoid the gymnasium dependency (same interface, zero extra deps)
    """

    # Space dimensions 
    OBS_DIM = 13
    ACT_DIM = 8

    # Reward wts.
    W_THROUGHPUT = 0.005    # per kW of IT load served
    W_VIOLATION  = 100.0   # per degC above 40 (squared penalty)
    W_ENERGY     = 0.001   # per kW of total facility power

    # Action bounds
    DELTA_TEMP_MAX = 2.0    # degC per timestep
    DELTA_FAN_MAX  = 0.10   # fraction per timestep

    # Setpoint bounds
    T_SUPPLY_MIN, T_SUPPLY_MAX = 6.0, 18.0
    FAN_MIN, FAN_MAX = 0.3, 1.0

    # Thermal hard lim
    T_OUTLET_LIMIT = 40.0   # degC

    def __init__(self, facility, seed: Optional[int] = None):
        """
        Parameters
        ----------
        facility : NovaCoolFacility instance
        seed     : Random seed (for any future stochastic extensions)
        """
        self.facility = facility
        self.rng = np.random.default_rng(seed)

        # Episode state
        self._t: int = 0
        self._crah_supply = np.full(4, 7.0)
        self._crah_fans   = np.ones(4)
        self._last_th: Optional[dict] = None
        self._last_pw: Optional[dict] = None



    # Core Gym interface

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial conditions.
        Returns initial observation.
        """
        self._t = 0
        self._crah_supply = np.full(4, 7.0)
        self._crah_fans   = np.ones(4)
        self._last_th     = None
        self._last_pw     = None
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Advance simulation by one 1-min timestep.

        Parameters
        ----------
        action : (8,) array
            [0:4] delta_supply_temp per CRAH [degC]
            [4:8] delta_fan_frac per CRAH [fraction]

        Returns
        -------
        obs    : (13,) observation vector
        reward : scalar reward signal
        done   : True when episode ends (t == 1440)
        info   : diagnostic dict (pue, violations, max_outlet, etc.)
        """
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (self.ACT_DIM,), (
            f"Action must be shape ({self.ACT_DIM},), got {action.shape}"
        )

        # applying action deltas
        delta_temp = np.clip(action[:4], -self.DELTA_TEMP_MAX, self.DELTA_TEMP_MAX)
        delta_fan  = np.clip(action[4:], -self.DELTA_FAN_MAX,  self.DELTA_FAN_MAX)

        self._crah_supply = np.clip(
            self._crah_supply + delta_temp,
            self.T_SUPPLY_MIN, self.T_SUPPLY_MAX
        )
        self._crah_fans = np.clip(
            self._crah_fans + delta_fan,
            self.FAN_MIN, self.FAN_MAX
        )

        # Step physics 
        rack_power = self.facility.workload[self._t]
        th = self.facility.thermal.step(
            rack_power, self._crah_supply, self._crah_fans
        )
        pw = self.facility.power.step(
            rack_power.sum(), th["crah_cooling_kw"], self._crah_fans
        )
        self._last_th = th
        self._last_pw = pw

        # find Reward
        reward = self._compute_reward(rack_power.sum(), th, pw)

        # Timestep + = 1
        self._t += 1
        done = self._t >= self.facility.N_STEPS

        obs  = self._get_obs(th, pw, rack_power.sum())
        info = {
            "t":           self._t,
            "pue":         pw["pue"],
            "max_outlet":  float(th["outlet_temp_c"].max()),
            "n_violations": int((th["outlet_temp_c"] > self.T_OUTLET_LIMIT).sum()),
            "total_it_kw": float(rack_power.sum()),
            "fan_power_kw": pw["fan_power_kw"],
            "crah_supply_temps": self._crah_supply.copy(),
            "crah_fan_fracs":    self._crah_fans.copy(),
        }

        return obs, reward, done, info


    # Reward function

    def _compute_reward(
        self,
        it_load_kw: float,
        th: dict,
        pw: dict,
    ) -> float:
        """
        Reward = throughput bonus - violation penalty - energy cost

        """
        max_outlet = float(th["outlet_temp_c"].max())

        throughput_bonus = self.W_THROUGHPUT * it_load_kw

        # Squared violation penalty ... zero below limit, and grows fast above
        violation = max(0.0, max_outlet - self.T_OUTLET_LIMIT)
        violation_penalty = self.W_VIOLATION * (violation ** 2)

        energy_cost = self.W_ENERGY * pw["total_facility_kw"]

        return throughput_bonus - violation_penalty - energy_cost

    # Observation builder
    
    def _get_obs(
        self,
        th: Optional[dict] = None,
        pw: Optional[dict] = None,
        it_load_kw: float = 0.0,
    ) -> np.ndarray:
        """
        Build normalised 13 dim observation vector,
        All values are scaled to approximately [0, 1] for RL stability.
        """
        if th is None:
            # Initial obsn before first step
            return np.array([
                8.5 / 20.0,    # cold aisle (supply+ recirculation)
                8.5 / 50.0,    # hot aisle
                8.5 / 50.0,    # max outlet
                0.0,           # IT load
                0.0,           # time of day
                *self._crah_supply / 20.0,
                *self._crah_fans,
            ], dtype=np.float64)

        return np.array([
            th["inlet_temp_c"].mean() / 20.0,
            th["hot_aisle_temp_c"].mean() / 50.0,
            th["outlet_temp_c"].max() / 50.0,
            it_load_kw / 10_000.0,
            self._t / self.facility.N_STEPS,
            *self._crah_supply / 20.0,
            *self._crah_fans,
        ], dtype=np.float64)

    # Space descriptors (Gym-compatible)
    
    @property
    def observation_space(self) -> dict:
        """Observation space metadata for RL engineers/team"""
        return {
            "shape": (self.OBS_DIM,),
            "low":   np.zeros(self.OBS_DIM),
            "high":  np.ones(self.OBS_DIM),
            "dtype": np.float64,
            "labels": [
                "avg_cold_aisle_temp_norm",
                "avg_hot_aisle_temp_norm",
                "max_outlet_temp_norm",
                "total_it_load_norm",
                "time_of_day_norm",
                "crah1_supply_temp_norm",
                "crah2_supply_temp_norm",
                "crah3_supply_temp_norm",
                "crah4_supply_temp_norm",
                "crah1_fan_frac",
                "crah2_fan_frac",
                "crah3_fan_frac",
                "crah4_fan_frac",
            ],
        }

    @property
    def action_space(self) -> dict:
        """Action space metadata for RL engineers or team"""
        return {
            "shape": (self.ACT_DIM,),
            "low":   np.array([-self.DELTA_TEMP_MAX]*4 + [-self.DELTA_FAN_MAX]*4),
            "high":  np.array([ self.DELTA_TEMP_MAX]*4 + [ self.DELTA_FAN_MAX]*4),
            "dtype": np.float64,
            "labels": [
                "delta_supply_temp_crah1",
                "delta_supply_temp_crah2",
                "delta_supply_temp_crah3",
                "delta_supply_temp_crah4",
                "delta_fan_frac_crah1",
                "delta_fan_frac_crah2",
                "delta_fan_frac_crah3",
                "delta_fan_frac_crah4",
            ],
        }

    def __repr__(self) -> str:
        return (
            f"DataCenterEnv("
            f"obs_dim={self.OBS_DIM}, "
            f"act_dim={self.ACT_DIM}, "
            f"t={self._t}/{self.facility.N_STEPS})"
        )