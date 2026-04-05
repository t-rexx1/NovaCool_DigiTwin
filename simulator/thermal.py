"""
Description:

Thermal model for NovaCool data center digital twin.

Design philosophy (Zohdi 2022):
    Each rack (of 50kW) is treated as a heat generating device in a coupled energy management system. We use a lumped-parameter approach (reduced order model(rom)) over CFD as suggested in problem statement for quick computation and interpretability.

Energy balance per rack (from FLOT):
    Q_gen[i] = mdot_air*cp_air * (T_outlet[i] - T_inlet[i])
    ==> T_outlet[i] = T_inlet[i] + Q_gen[i] / (mdot_air * cp_air)

CRAH: Computer Room Air Handler unit.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThermalConfig:
    """Physical parameters for NovaCool facility thermal model"""
    n_racks: int = 200
    n_crahs: int = 4
    racks_per_crah: int = 50          # 4 rows x 50 racks, one CRAH per row

    # Air properties
    mdot_air_per_rack: float = 0.36   # kg/s (rho=1.2 kg/m3  x  Vdot=0.3 m3/s)
    cp_air: float = 1006.0            # J/(kg.K)

    # CRAH defaults
    crah_supply_temp_default: float = 7.0   # Celsius
    crah_fan_frac_default: float = 1.0      # 0-1 (frac. of rated speed)

    # Recirculation: hot aisle air leaking back to cold aisle which accounts for bypass airflow and is penalized
    recirculation_factor: float = 1.5       # this much deg C added to cold aisle inlet flow

    # Thermal constraint
    outlet_temp_limit: float = 40.0         # in Celsius, this is HARD/Max limit, can't exceed at any cost


class ThermalModel:
    """
    Lumped parameter thermal model of NovaCool data center.

    Models each rack as an individual heat source. Cold air supplied by CRAH units enters the cold aisle, absorbs heat passing through server racks, and then exits into the hot aisle. 
    
    CRAH units then cool the return air using a water chilled loop...

    Assumptions taken:
    - Steady state (SS) within each 1-min timestep (thermal mass << timestep)
    - Well mixed hot aisle for every/per CRAH zone
    - Uniform airflow dist. across all servers in a rack
    - Chilled water supply temperature is held constant (but is controllable ifneeded)
    - Recirculation modeled as a fixed temp. offset on cold aisle inlet
    """

    def __init__(self, config: Optional[ThermalConfig] = None):
        self.cfg = config or ThermalConfig()
        # Assigning CRAH index for each rack (since fixed layout)
        self._crah_idx = np.arange(self.cfg.n_racks) // self.cfg.racks_per_crah

    def step(self,
        rack_power_kw: np.ndarray,      # shape (n_racks,)  kW
        crah_supply_temps: np.ndarray,  # shape (n_crahs,)  Celsius
        crah_fan_fracs: np.ndarray,     # shape (n_crahs,)  0.0 to 1.0 frac.
    ) -> dict:
        """
        Compute thermal state for one(single) 1 minute timestep.

        Parameters
        ----------
        rack_power_kw   : Per rack IT power draw [kW]
        crah_supply_temps: Supply air temp per CRAH unit [deg C]
        crah_fan_fracs  : Fan speed fraction per CRAH (this affects airflow) [b/w 0 to 1]

        Returns
        -------
        dictionary with keys:
            inlet_temp_c     : Cold aisle inlet temp per rack  (n_racks,)
            outlet_temp_c    : Hot aisle outlet temp per rack  (n_racks,)
            hot_aisle_temp_c : Average hot aisle temp per CRAH (n_crahs,)
            crah_return_temp_c: Return air temp seen by each CRAH (n_crahs,)
            crah_cooling_kw  : Heat removed by each CRAH unit  (n_crahs,)
        """
        rack_power_w = rack_power_kw * 1000.0  # this is Watts instead of kWs

        
        # Cold aisle inlet temperature per rack
        # Each rack gets supply air from CRAH assigned to it + recirculation penalty...
        
        T_inlet = (crah_supply_temps[self._crah_idx] + self.cfg.recirculation_factor)


        # Effective mass flow rate scales with fan speed (comes from affinity law) 
        # So @ 50% fan speed, airflow drops to 50% .... means less cooling cap.
        mdot = np.clip(
            self.cfg.mdot_air_per_rack * crah_fan_fracs[self._crah_idx],
            a_min=0.05,   # never fully zero to prevent div by 0 errs
            a_max=None,
        )

        # Rack outlet temperature (FLOT)
        # Q = mdot * cp * delta_T  ==>  T_out = T_in + Q / (mdot * cp)
        T_outlet = T_inlet + rack_power_w / (mdot * self.cfg.cp_air)

        # Hot aisle temperature per CRAH zone
        T_hot_aisle = np.array([
            T_outlet[self._crah_idx == j].mean()
            for j in range(self.cfg.n_crahs)
        ])

        # CRAH return temperature (is= hot aisle, well mixed assumption!)
        T_return = T_hot_aisle.copy()

        # Heat removed by each CRAH [in kW]
        # Q_crah = mdot_total * cp * (T_return - T_supply)
        mdot_total = (
            self.cfg.mdot_air_per_rack
            * self.cfg.racks_per_crah
            * crah_fan_fracs
        )
        Q_removed_kw = (
            mdot_total * self.cfg.cp_air
            * (T_return - crah_supply_temps)
            / 1000.0
        )
        Q_removed_kw = np.clip(Q_removed_kw, a_min=0.0, a_max=None)

        return {
            "inlet_temp_c":      T_inlet,       # (n_racks,)
            "outlet_temp_c":     T_outlet,      # (n_racks,)
            "hot_aisle_temp_c":  T_hot_aisle,   # (n_crahs,)
            "crah_return_temp_c": T_return,     # (n_crahs,)
            "crah_cooling_kw":   Q_removed_kw,  # (n_crahs,)
        }

    def n_violations(self, outlet_temps: np.ndarray) -> int:
        """Count racks exceeding the 40 degC outlet temperature limit."""
        return int((outlet_temps > self.cfg.outlet_temp_limit).sum())