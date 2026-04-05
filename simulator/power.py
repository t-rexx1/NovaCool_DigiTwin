"""
Power model for NovaCool: UPS efficiency, cooling infrastructure power, PUE.

UPS efficiency curve (quadratic, peaks at around ~62.5% load):
    eta(x) = eta_peak - k*(x - x_peak)^2
    where x = IT_load/capacity

Fan power (cube law from affinity laws):
    P_fan = P_rated * fan_frac^3

PUE = Total Facility Power / IT Load
    Target: < 1.5 (good), < 1.2 (excellent)

UPS: Uninterruptable Power Supply,
PUE: Power Usage Effectiveness
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class PowerConfig:
    """Electrical parameters for NovaCool..."""
    it_capacity_kw: float = 10_000.0      # Total IT capacity

    # UPS quadratic efficiency model
    ups_peak_efficiency: float = 0.97     # peak efficiency
    ups_peak_load_frac: float = 0.625     # load fraction at peak efficiency
    ups_quadratic_k: float = 0.23         # curvature

    # CRAH fan power
    fan_rated_kw_per_crah: float = 50.0   # kW at full speed

    # Chilled water pump: linear with cooling load
    pump_kw_per_kw_cooling: float = 0.02  # 2% of cooling power

    # Lighting / misc overhead
    overhead_kw: float = 50.0


class PowerModel:
    """
    Models the electrical power path and facility PUE for NovaCool.

    Key coupling with thermal model:
    - IT load drives both heat generation AND the UPS losses
    - Cooling load drives CRAH fan + pump power
    - Higher fan speed -> more cooling but higher P_fan (cube law)
    """

    def __init__(self, config: Optional[PowerConfig] = None):
        self.cfg = config or PowerConfig()

    def ups_efficiency(self, it_load_kw: float) -> float:
        """
        Quadratic UPS efficiency model.
        Peaks at cfg.ups_peak_load_frac of rated capacity.
        Drops off at very low or very high loads.
        """
        x = it_load_kw / self.cfg.it_capacity_kw
        eta = (
            self.cfg.ups_peak_efficiency
            - self.cfg.ups_quadratic_k * (x - self.cfg.ups_peak_load_frac) ** 2
        )
        return float(np.clip(eta, 0.5, 1.0))  # physical bounds

    def step(
        self,
        it_load_kw: float,
        crah_cooling_kw: np.ndarray,  # (n_crahs,)
        crah_fan_fracs: np.ndarray,   # (n_crahs,)
    ) -> dict:
        """
        Compute facility power budget for one timestep.

        Returns
        -------
        dict. with keys:
            ups_efficiency    : UPS efficiency at current load [0-1]
            ups_input_kw      : Power drawn from grid for IT [kW]
            fan_power_kw      : Total CRAH fan power [kW]
            pump_power_kw     : Chilled-water pump power [kW]
            overhead_kw       : Lighting + misc [kW]
            total_facility_kw : Total power drawn from grid [kW]
            pue               : Power Usage Effectiveness [-]
        """
        # UPS input = IT load / efficiency (losses appear as heat)
        eta = self.ups_efficiency(it_load_kw)
        ups_input_kw = it_load_kw / max(eta, 0.1)

        # Fan power: cube law (doubling airflow -> 8x power)
        fan_power_kw = (
            self.cfg.fan_rated_kw_per_crah
            * np.sum(crah_fan_fracs ** 3)
        )

        # Pump power: proportional to total cooling load
        pump_power_kw = self.cfg.pump_kw_per_kw_cooling * crah_cooling_kw.sum()

        total_facility_kw = (
            ups_input_kw
            + fan_power_kw
            + pump_power_kw
            + self.cfg.overhead_kw
        )

        pue = total_facility_kw / max(it_load_kw, 1.0)

        return {
            "ups_efficiency":    eta,
            "ups_input_kw":      ups_input_kw,
            "fan_power_kw":      fan_power_kw,
            "pump_power_kw":     pump_power_kw,
            "overhead_kw":       self.cfg.overhead_kw,
            "total_facility_kw": total_facility_kw,
            "pue":               pue,
        }