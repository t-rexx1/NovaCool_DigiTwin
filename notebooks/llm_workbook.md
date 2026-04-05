# LLM Reasoning Workbook - NovaCool Digital Twin
**Author:** Aaditya Shrivastava
**Tool used:** Claude (claude.ai chat interface + CodeX via Berkeley AI Sandbox)
**Purpose:** Documents where LLM assistance was used, what was suggested, and how I evaluated and applied (or rejected) each suggestion.

---

## Q1 - Mass flow rate for rack cooling air

**What I asked:** What mass flow rate is appropriate for a 50 kW server rack in a hot-aisle/cold-aisle data center?

**LLM suggestion:** 0.36 kg/s based on rho=1.2 kg/m3 * Vdot=0.3 m3/s

**My evaluation and decision:**
Ran the simulation with this value and got outlet temperatures of 132 degC, which is physically impossible. The first-law equation immediately revealed the bug:

T_outlet = T_inlet + Q / (mdot * cp)
= 8.5 + 45,000 / (0.36 * 1006) = 132 degC (wrong)

Back-calculated the correct value from sensor_reference.csv:

mdot = Q / (cp * dT) = 45,000 / (1006 * 28.5) ~ 1.57 kg/s

Set to 1.5 kg/s (conservative, within ASHRAE A2 envelope for high-density racks where Vdot ~ 1.25 m3/s). This reduced peak outlet from 132 degC to 38.3 degC with 0 violations, which is physically correct.

**Lesson:** The LLM gave a plausible-sounding but wrong parameter. Physics caught it immediately. Always validate model parameters against actual outputs.

---

## Q2 - Reward function design for the RL environment

**What I asked:** How should I design the reward function for a data center thermal control RL environment?

**LLM suggestion:** Linear violation penalty with weight 50x, throughput bonus, energy cost penalty.

**My evaluation and decision:**
Rejected the linear penalty in favour of a squared violation penalty (weight 100x):
python
violation_penalty = W_VIOLATION * max(0, T_max - 40.0) ** 2


Reason: a linear penalty creates a flat gradient, so the agent learns to hover exactly at the 40 degC boundary (boundary-hugging behaviour), which is unsafe in real deployment. A squared term creates an accelerating gradient that pushes the agent to stay well clear of the limit.

Weight ratio chosen: w_violation/w_throughput = 100/0.005 = 20,000. This ensures the agent can never profitably trade thermal violations for throughput - safety is non-negotiable.

**Accepted:** Three-term reward structure (throughput - violation - energy).
**Modified:** Linear -> squared violation, weight 50 -> 100.

---

## Q3 - Episode termination on violation?

**What I asked:** Should the episode terminate early when a thermal violation occurs?

**LLM suggestion:** Terminate early on critical violation (outlet > 40 degC)

**My evaluation and decision:**
Rejected. Early termination on violation causes two problems:
1. Truncated episodes create biased Q-value estimates at terminal states (the agent sees a false "end of world" signal rather than an ongoing cost)
2. The agent gets less training signal about what happens after a violation

Decision: full 1,440-step episodes always. The squared penalty provides sufficient signal - the agent experiences the ongoing cost of violations rather than just episode death. This produces more stable training.

---

## Q4 - Calibration approaches for Task 3

**What I asked:** What calibration methods should I discuss for improving model fidelity?

**LLM suggestion:** Bayesian parameter tuning and Kalman filter assimilation

**My evaluation:**
Both are well-motivated for different use cases:

- **Bayesian (offline):** Treat mdot and recirculation_factor as uncertain priors, update posteriors using MCMC against historical sensor data. Best during commissioning when you have batch data but no live stream.

- **Kalman filter (online):** At each timestep, blend model prediction with live sensor reading: x+ = x- + K(y - Hx-). This corrects slow drift from parameter changes (e.g., filter fouling changes airflow resistance) without full recalibration. This is what Prof Zohdi (2022) calls sim-to-real alignment in a running digital twin.

**Accepted both** - they address different phases of the system lifecycle.

---

## Q5 - Heuristic controller violations (true finding)

**Observation during results analysis:**
The heuristic controller produced 10 thermal violations vs 0 for baseline, despite a better mean PUE (1.101 vs 1.134).

**Root cause (my analysis, not LLM):**
The dead-band thermostat is purely reactive - it raises supply temperature during low load periods to save energy, but cannot anticipate the sharp workload ramp at 05:00. By the time the controller detects T_max > T_high and starts cooling down, the thermal lag means 10 racks briefly exceed 40 degC.

**Why this validates the RL motivation:**
A trained RL agent observing time of day in its observation vector could learn to preemptively lower supply temperature before the 05:00 ramp.

This is the core value proposition and advantage of the DataCenterEnv - to be able to use agents that are predictive and not just reactive.

**This finding is left in the results intentionally** - to represent true analysis
