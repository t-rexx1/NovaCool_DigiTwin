"""
This generates all 4 (see edit, now 5) plots.
Run from repo root: python notebooks/generate_plots.py

Edit: Added 1 more plot for CRAH failure perturbation (stretch goal) at the end of the script.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.facility import NovaCoolFacility
from control.baseline import FixedSetpointController
from control.heuristic import ThermostatController
from validation.metrics import run_validation, load_sensor_reference, sensor_to_arrays

# Setup 
plt.rcParams.update({
    "figure.dpi": 120, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUT = Path("writeup")
OUT.mkdir(exist_ok=True)
DATA = Path("data")

print("Loading facility...")
facility = NovaCoolFacility.from_csvs(DATA / "workload_trace.csv")

# Run all scenarios
print("Running baseline...")
b_supply, b_fans = FixedSetpointController().run_episode(facility)
results_baseline = facility.run(b_supply, b_fans)

print("Running heuristic controller...")
heuristic_ctrl = ThermostatController()
h_supply, h_fans = heuristic_ctrl.run_episode(facility)
results_heuristic = facility.run(h_supply, h_fans)

print("Running validation...")
val = run_validation(facility, DATA / "sensor_reference.csv")
sensor_arrays = sensor_to_arrays(load_sensor_reference(DATA / "sensor_reference.csv"))
ref_outlet_mean = sensor_arrays["outlet_temp_c"].mean(axis=1)

t_hours = np.arange(1440) / 60.0

# Figure 1: Part A required time series 
print("Generating Figure 1...")
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
fig.suptitle(
    "NovaCool Digital Twin — 24-Hour Simulation (Baseline)\n"
    "Fixed CRAH setpoints: 7°C supply, 100% fan speed",
    fontsize=13, fontweight="bold", y=0.98
)

ax = axes[0]
ax.plot(t_hours, results_baseline["avg_cold_aisle_c"],
        color="#2196F3", linewidth=1.5, label="Simulated (baseline)")
ax.axhline(7.0, color="gray", linestyle="--", linewidth=1, label="CRAH supply (7°C)")
ax.set_ylabel("Avg Cold-Aisle Temp (°C)")
ax.set_title("(a) Cold-Aisle Temperature")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 20)

ax = axes[1]
ax.plot(t_hours, results_baseline["avg_hot_aisle_c"],
        color="#F44336", linewidth=1.5, label="Simulated hot aisle")
ax.axhline(35.0, color="orange", linestyle="--", linewidth=1,
           label="Return target (35°C)")
ax.set_ylabel("Avg Hot-Aisle Temp (°C)")
ax.set_title("(b) Hot-Aisle Temperature")
ax.legend(loc="upper right", fontsize=9)

ax = axes[2]
ax.plot(t_hours, results_baseline["peak_outlet_c"],
        color="#9C27B0", linewidth=1.5, label="Peak rack outlet (simulated)")
ax.plot(t_hours, ref_outlet_mean,
        color="#9C27B0", linewidth=1, linestyle="--",
        alpha=0.6, label="Mean outlet (sensor reference)")
ax.axhline(40.0, color="red", linestyle="-", linewidth=1.5,
           alpha=0.7, label="Hard limit (40°C)")
ax.set_ylabel("Rack Outlet Temp (°C)")
ax.set_title("(c) Peak Rack Outlet Temperature vs Sensor Reference")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 50)

ax = axes[3]
ax.plot(t_hours, results_baseline["total_cooling_kw"],
        color="#4CAF50", linewidth=1.5, label="CRAH cooling power")
ax.fill_between(t_hours, results_baseline["total_cooling_kw"],
                alpha=0.15, color="#4CAF50")
ax.set_ylabel("Cooling Power (kW)")
ax.set_xlabel("Time of Day (hours)")
ax.set_title("(d) Total Cooling Power Consumed")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))

plt.tight_layout()
plt.savefig(OUT / "fig1_partA_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: writeup/fig1_partA_timeseries.png")

# Figure 2: PUE profile
print("Generating Figure 2...")
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle("Power Usage Effectiveness (PUE) — 24-Hour Profile",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(t_hours, results_baseline["total_it_kw"],
        color="#2196F3", linewidth=1.5, label="IT load (kW)")
ax.fill_between(t_hours, results_baseline["total_it_kw"],
                alpha=0.1, color="#2196F3")
ax.set_ylabel("Total IT Load (kW)")
ax.set_title("(a) IT Load Profile")
ax.legend(loc="upper right", fontsize=9)

ax = axes[1]
ax.plot(t_hours, results_baseline["pue"],
        color="#FF9800", linewidth=1.5, label="PUE (baseline)")
ax.axhline(results_baseline["pue"].mean(), color="#FF9800",
           linestyle="--", linewidth=1,
           label=f"Mean PUE = {results_baseline['pue'].mean():.3f}")
ax.axhline(1.2, color="green", linestyle=":", linewidth=1.5,
           label="World-class target (1.2)")
ax.axhline(2.0, color="red", linestyle=":", linewidth=1,
           label="Industry average (2.0)")
ax.set_ylabel("PUE (-)")
ax.set_xlabel("Time of Day (hours)")
ax.set_title("(b) Power Usage Effectiveness")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(1.0, 1.5)
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))

plt.tight_layout()
plt.savefig(OUT / "fig2_pue.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: writeup/fig2_pue.png")

# Figure 3: Validation 
print("Generating Figure 3...")
timestep_errors = val["timestep_errors"]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Model Validation — Simulated vs Sensor Reference",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(t_hours, results_baseline["peak_outlet_c"],
        color="#9C27B0", linewidth=1.5, label="Simulated peak outlet")
ax.plot(t_hours, ref_outlet_mean,
        color="#9C27B0", linewidth=1, linestyle="--", alpha=0.7,
        label="Sensor reference mean outlet")
ax.fill_between(t_hours,
                results_baseline["peak_outlet_c"], ref_outlet_mean,
                alpha=0.2, color="#9C27B0", label="Error region")
ax.axhline(40.0, color="red", linestyle="-", linewidth=1.5,
           alpha=0.6, label="Hard limit (40°C)")
m = val["metrics_outlet"]
ax.set_ylabel("Outlet Temperature (°C)")
ax.set_title(
    f"(a) Outlet Temp: Simulated vs Reference  "
    f"[RMSE={m['rmse']:.2f}°C  MAE={m['mae']:.2f}°C  "
    f"MaxAE={m['max_absolute_error']:.2f}°C]"
)
ax.legend(loc="upper right", fontsize=9)

ax = axes[1]
ax.plot(t_hours, timestep_errors,
        color="#F44336", linewidth=1.2, label="MAE per timestep")
ax.axhline(np.percentile(timestep_errors, 90),
           color="orange", linestyle="--", linewidth=1.5,
           label="90th percentile threshold")
labeled = False
for w in val["divergence_windows"]:
    lbl = f"Divergence window" if not labeled else "_nolegend_"
    ax.axvspan(w["start_min"] / 60, w["end_min"] / 60,
               alpha=0.25, color="red", label=lbl)
    ax.text((w["start_min"] + w["end_min"]) / 2 / 60,
            timestep_errors.max() * 0.92,
            f"{w['start_hhmm']}\n–{w['end_hhmm']}",
            ha="center", fontsize=8, color="darkred")
    labeled = True

ax.set_ylabel("MAE per Timestep (°C)")
ax.set_xlabel("Time of Day (hours)")
ax.set_title("(b) Per-Timestep Error — Divergence Window Analysis")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))

plt.tight_layout()
plt.savefig(OUT / "fig3_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: writeup/fig3_validation.png")

# Figure 4: Control comparison
print("Generating Figure 4...")
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
fig.suptitle("Control Policy Comparison: Baseline vs Heuristic Thermostat",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(t_hours, results_baseline["peak_outlet_c"],
        color="#F44336", linewidth=1.5, alpha=0.8,
        label="Baseline (fixed 7°C supply)")
ax.plot(t_hours, results_heuristic["peak_outlet_c"],
        color="#2196F3", linewidth=1.5, alpha=0.8,
        label="Heuristic (dead-band thermostat)")
ax.axhline(40.0, color="red", linestyle="-", linewidth=2,
           alpha=0.5, label="Hard limit (40°C)")
ax.axhline(37.0, color="orange", linestyle="--", linewidth=1,
           alpha=0.7, label="T_high dead-band (37°C)")
ax.set_ylabel("Peak Outlet Temp (°C)")
ax.set_title("(a) Thermal Safety — Peak Rack Outlet Temperature")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 45)

ax = axes[1]
ax.plot(t_hours, results_baseline["pue"],
        color="#F44336", linewidth=1.5, alpha=0.8,
        label=f"Baseline  (mean PUE={results_baseline['pue'].mean():.3f})")
ax.plot(t_hours, results_heuristic["pue"],
        color="#2196F3", linewidth=1.5, alpha=0.8,
        label=f"Heuristic (mean PUE={results_heuristic['pue'].mean():.3f})")
ax.set_ylabel("PUE (-)")
ax.set_title("(b) Energy Efficiency — Power Usage Effectiveness")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(1.0, 1.5)

ax = axes[2]
ax.plot(t_hours, h_supply[:, 0], linewidth=1.2, label="CRAH 1")
ax.plot(t_hours, h_supply[:, 1], linewidth=1.2,
        linestyle="--", label="CRAH 2")
ax.plot(t_hours, h_supply[:, 2], linewidth=1.2,
        linestyle=":", label="CRAH 3")
ax.plot(t_hours, h_supply[:, 3], linewidth=1.2,
        linestyle="-.", label="CRAH 4")
ax.axhline(7.0, color="gray", linestyle=":", linewidth=1,
           label="Baseline fixed setpoint (7°C)")
ax.set_ylabel("Supply Temp (°C)")
ax.set_xlabel("Time of Day (hours)")
ax.set_title("(c) Heuristic Controller — CRAH Supply Temperature Schedule")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))

plt.tight_layout()
plt.savefig(OUT / "fig4_control_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: writeup/fig4_control_comparison.png")

# Task 4 comparison table 
print("\n=== Task 4 Performance Comparison ===")
it_b_mwh   = results_baseline["total_it_kw"].sum() / 60 / 1000
it_h_mwh   = results_heuristic["total_it_kw"].sum() / 60 / 1000
cool_b_mwh = results_baseline["total_cooling_kw"].sum() / 60 / 1000
cool_h_mwh = results_heuristic["total_cooling_kw"].sum() / 60 / 1000
pue_b = results_baseline["pue"].mean()
pue_h = results_heuristic["pue"].mean()
viol_b = results_baseline["n_violations"].sum()
viol_h = results_heuristic["n_violations"].sum()

print(f"{'Metric':<35} {'Baseline':>12} {'Heuristic':>12}")
print("-" * 60)
print(f"{'Total IT Load Served (MWh)':<35} {it_b_mwh:>12.2f} {it_h_mwh:>12.2f}")
print(f"{'Thermal Violations':<35} {viol_b:>12} {viol_h:>12}")
print(f"{'Total Cooling Energy (MWh)':<35} {cool_b_mwh:>12.2f} {cool_h_mwh:>12.2f}")
print(f"{'Mean PUE':<35} {pue_b:>12.3f} {pue_h:>12.3f}")
pue_improvement = (pue_b - pue_h) / pue_b * 100
print(f"{'PUE Improvement (%)':<35} {'—':>12} {pue_improvement:>11.1f}%")
print(f"\nAll 5 figures saved to writeup/")


# Fig 5: CRAH Failure Perturbation (Stretch Goal) 
print("Generating Fig 5 CRAH failure perturbation...")

# Simulate CRAH 0 failing from t=300 to t=480 (05:00–08:00)
# During failure: CRAH 0 fan drops to 0, supply temp rises to ambient
FAIL_START, FAIL_END = 300, 480

shape = (facility.N_STEPS, 4)
T_supply_fail = np.full(shape, 7.0)
fan_fail       = np.ones(shape)

# Injecting failure: CRAH 0 loses cooling capacity
T_supply_fail[FAIL_START:FAIL_END, 0] = 35.0  # supply warms to hot aisle return
fan_fail[FAIL_START:FAIL_END, 0]       = 0.3   # fans at minimum

results_fail = facility.run(T_supply_fail, fan_fail)

# Heuristic response to failure
heuristic_fail = ThermostatController()
h_supply_fail = np.full(shape, 7.0)
h_fan_fail    = np.ones(shape)
h_supply_fail[FAIL_START:FAIL_END, 0] = 35.0
h_fan_fail[FAIL_START:FAIL_END, 0]    = 0.3
h_s, h_f = heuristic_fail.run_episode(
    type('obj', (object,), {
        'workload': facility.workload,
        'N_STEPS': facility.N_STEPS,
        'thermal': facility.thermal,
    })()
)
# Override failed CRAH
h_s[FAIL_START:FAIL_END, 0] = 35.0
h_f[FAIL_START:FAIL_END, 0] = 0.3
results_fail_heuristic = facility.run(h_s, h_f)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(
    "Stretch Goal: CRAH 0 Failure Injection (05:00–08:00)\n"
    "Baseline vs Heuristic Response",
    fontsize=13, fontweight="bold"
)

ax = axes[0]
ax.axvspan(FAIL_START/60, FAIL_END/60, alpha=0.15, color="red",
           label="CRAH 0 failure window")
ax.plot(t_hours, results_baseline["peak_outlet_c"],
        color="gray", linewidth=1, linestyle="--", alpha=0.6,
        label="No failure (baseline)")
ax.plot(t_hours, results_fail["peak_outlet_c"],
        color="#F44336", linewidth=1.5,
        label="Fixed setpoint — failure")
ax.plot(t_hours, results_fail_heuristic["peak_outlet_c"],
        color="#2196F3", linewidth=1.5,
        label="Heuristic — failure response")
ax.axhline(40.0, color="red", linestyle="-", linewidth=2,
           alpha=0.5, label="Hard limit (40°C)")
ax.set_ylabel("Peak Outlet Temp (°C)")
ax.set_title("(a) Thermal Response to CRAH 0 Failure")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 55)

ax = axes[1]
ax.axvspan(FAIL_START/60, FAIL_END/60, alpha=0.15, color="red")
ax.plot(t_hours, results_baseline["pue"],
        color="gray", linewidth=1, linestyle="--", alpha=0.6,
        label="No failure (baseline)")
ax.plot(t_hours, results_fail["pue"],
        color="#F44336", linewidth=1.5, label="Fixed setpoint — failure")
ax.plot(t_hours, results_fail_heuristic["pue"],
        color="#2196F3", linewidth=1.5, label="Heuristic — failure response")
ax.set_ylabel("PUE (-)")
ax.set_xlabel("Time of Day (hours)")
ax.set_title("(b) PUE Impact During CRAH Failure")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(1.0, 2.0)
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))

plt.tight_layout()
plt.savefig(OUT / "fig5_crah_failure.png", dpi=150, bbox_inches="tight")
plt.close()

# Print failure stats
viol_fail = results_fail["n_violations"].sum()
viol_fail_h = results_fail_heuristic["n_violations"].sum()
peak_fail = results_fail["peak_outlet_c"].max()
peak_fail_h = results_fail_heuristic["peak_outlet_c"].max()
print(f"\n=== CRAH Failure Results ===")
print(f"{'Metric':<35} {'Fixed':>10} {'Heuristic':>10}")
print("-" * 56)
print(f"{'Thermal violations':<35} {viol_fail:>10} {viol_fail_h:>10}")
print(f"{'Peak outlet temp (°C)':<35} {peak_fail:>10.2f} {peak_fail_h:>10.2f}")
print("  Saved: writeup/fig5_crah_failure.png")