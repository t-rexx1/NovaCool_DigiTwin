# NovaCool Digital Twin Architecture

Physics based (lump mass) digital twin of a 10 MW colocation data center, with 4 racks of 50kW servers each,
built for the Hammerhead AI Sim. Engineer case study.

## Approach

I drew inspirstion from the lumped parameter digital twin methodology of Prof. Tarek Zohdi descrived in three pf his publications (2022[2] and 2023), each rack is modeled as a heat generating device in a coupled energy management system. The simulation runs at 1-minute timesteps over a 24 hr period and is exposed as a Gym style Reinforcement learning training environment.

## Structure

    simulator/   # Thermal + power physics models
    control/     # Baseline + heuristic controllers
    env/         # DataCenterEnv (Gym style RL interface)
    validation/  # RMSE/MAE metrics vs sensor reference
    notebooks/   # Results, plots, LLM assuming/reasoning workbook
    writeup/     # LaTeX PDF write up
    tests/       # Unit tests for the physics eqns...

## Setup

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run

    python -m simulator.facility

## Data

Place workload_trace.csv and sensor_reference.csv in data/ folder

## References

T.I. Zohdi, A machine-learning digital-twin for rapid large-scale solar-thermal energy system design, Computer Methods in Applied Mechanics and Engineering, Volume 412, 2023, 115991, ISSN 0045-7825, https://doi.org/10.1016/j.cma.2023.115991.

Zohdi, T.I. A digital-twin and machine-learning framework for precise heat and energy management of data-centers. Comput Mech 69, 1501–1516 (2022). https://doi.org/10.1007/s00466-022-02152-3

Zohdi, T.I. An adaptive digital framework for energy management of complex multi-device systems. Comput Mech 70, 867–878 (2022). https://doi.org/10.1007/s00466-022-02212-8