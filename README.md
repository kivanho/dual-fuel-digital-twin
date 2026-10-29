# Dual-Fuel Engine Digital Twin

A digital twin of a diesel/methane dual-fuel generator set (16.5 kVA / 13.2 kWe prime, 18 kVA / 14.4 kWe standby: a 2.2 L four-cylinder diesel at 1500 rpm driving a three-phase 400 V alternator), served as a FastAPI service in Docker.
You set the operating point (electrical power, diesel energy share, methane content of the gas), and the twin predicts the target efficiency and exhaust temperature, works out the fuel mass flows from an energy balance, and lets two PI flow loops drive the simulated fuel valves towards those targets in real time.

Master's project at IMT Atlantique (Feb–Jun 2025). Team of three; I led the project and did the main implementation.
This repository is a cleaned re-publication of that code (2026): the original repository stays private because it contains the lab data.

> **About the data.** The original models were trained on lab measurements from the test bench, which are not public.
> This repository ships a **synthetic engine map** (`app/synthetic_data.py`) with the same columns and physically plausible trends, so everything runs end to end.
> At 100 % CH4 and 30 % diesel energy share its efficiency and exhaust-temperature curves are anchored to the diesel-only genset datasheet (efficiency within 1 percentage point at 50, 75 and 100 % load). Treating the dual-fuel engine like the diesel original at that point is an assumption; the dual-fuel effects on top are illustrative, not measurements of the real engine.
>
> **Model scores.** On the lab data, the kNN efficiency model reached R² 0.93 on a 20 % hold-out set and the exhaust-temperature regression R² 0.96 (training fit). On the synthetic map, `python -m scripts.train_models` prints R² 0.94 (kNN, hold-out) and 0.99 (exhaust). The 0.99 says nothing about real engines: the synthetic exhaust temperature is itself a near-linear formula.

## Architecture

```
                 POST /update_inputs
                 power [kW], diesel energy share [%], CH4 share [%]
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ SimulationService (one step per second, background asyncio task)  │
 │                                                                     │
 │  kNN regression ──► target electrical efficiency                    │
 │  (k=10, Manhattan; features: power, CH4 share)                      │
 │          │                                                          │
 │          ▼                                                          │
 │  Energy balance ──► target diesel and CH4 mass flows [kg/h]         │
 │  Q_fuel = P / η · 3.6,  split by diesel energy share, ÷ LHV         │
 │          │                                                          │
 │          ▼                                                          │
 │  PI flow loops + first-order valve model (anti-windup)              │
 │          │  actual fuel flows                                       │
 │          ▼                                                          │
 │  Plant outputs ──► power, diesel share, current, torque, CO2 in gas │
 │  Linear regression ──► exhaust temperature (target and actual)      │
 └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
          GET /live (snapshot)   GET /live_state (Server-Sent Events)
```

| File | Role |
|------|------|
| `app/main.py` | FastAPI app, background simulation loop, endpoints |
| `app/simulation_service.py` | Holds models, flow loops and state; one `step()` per second, JSON output |
| `app/simulation_core.py` | One simulation step: predict, balance, regulate, derive outputs from the actual flows |
| `app/knnr_model.py` | kNN efficiency model with a hold-out score |
| `app/exhaust_temp_model.py` | Linear exhaust-temperature model |
| `app/calculate_massflows.py` | Energy balance from power and efficiency to fuel mass flows |
| `app/regulator.py` | PI controller on a first-order valve model, with anti-windup |
| `app/engine_specs.py` | Genset ratings from the datasheet (voltage, cos φ, speed, power) |
| `app/synthetic_data.py` | Synthetic engine map, or your own CSV via `ENGINE_MAP_CSV` |
| `scripts/train_models.py` | Prints test-set metrics for both models, can export the map |

Both models train at startup in well under a second, so no model files are stored in the repo.

## Run with Docker

```bash
docker build -t dual-fuel-twin .
docker run --rm -p 8000:8000 dual-fuel-twin
```

Interactive API docs: http://localhost:8000/docs

### Example calls

Set an operating point (12 kW, 20 % diesel energy share, 80 % CH4 in the gas):

```bash
curl -X POST http://localhost:8000/update_inputs \
  -H "Content-Type: application/json" \
  -d '{"power_output": 12, "diesel_energy_share": 20, "methane_percentage": 80}'
```

Read the current state once:

```bash
curl http://localhost:8000/live
```

Stream the state as Server-Sent Events (one event per second; drop `max_events` to stream indefinitely):

```bash
curl -N "http://localhost:8000/live_state?max_events=5"
```

Reset the controllers and close the simulated valves:

```bash
curl -X POST http://localhost:8000/reset
```

Ten seconds later the loops have settled (within 2 % after about 3 s for a step from 8 to 12 kW; about 5 s from standstill, where the CH4 valve saturates at first):

```json
{
  "time_s": 10.0,
  "input":  {"power": 12.0, "des_percent": 20.0, "ch4_percent": 80.0},
  "target": {"efficiency": 0.289, "diesel_mass_flow_kg_h": 0.701, "ch4_mass_flow_kg_h": 2.394, "exhaust_temp": 319.8},
  "actual": {"power": 11.99, "diesel_mass_flow_kg_h": 0.701, "ch4_mass_flow_kg_h": 2.39, "diesel_energy_share": 20.0,
             "co2_in_fuel_gas_kg_h": 1.639, "exhaust_temp": 319.5, "current": 21.6, "torque": 95.4}
}
```

Inputs must lie inside the engine map: 2–14.4 kW, 10–60 % diesel energy share, 55–100 vol-% CH4. Anything outside is rejected with HTTP 422 rather than extrapolated.

## Run locally

Python 3.11 or 3.12:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
pytest
python -m scripts.train_models
```

To train on your own measurements, point `ENGINE_MAP_CSV` at a CSV with the columns listed in `app/synthetic_data.py`.

## Scope and limitations

- **Inputs are set by hand** through `/update_inputs`. A live connection to the engine's sensors was planned as future work and is not part of this project; the genset's control panel exposes a CAN data link that such a connection could read from.
- The kNN model only uses power and CH4 share as features; diesel energy share enters through the energy balance and the exhaust-temperature model.
- Current is the three-phase line current at 400 V and cos φ 0.8 (I = P / (√3 · U · cos φ)). Torque is the engine shaft torque at 1500 rpm, T = P_el / (η · ω), with a constant shaft-to-terminal efficiency η = 14.4 / 18.0 = 0.8 from the datasheet; at the standby rating this gives 115 Nm against 114 Nm from the datasheet's mean effective pressure.
- Power is limited to the 14.4 kWe standby rating.
- During transients the electrical efficiency is held at the target operating point's value; power follows the actual fuel energy.
- CO2 in the fuel gas is derived from the CH4 volume share of the CH4/CO2 mix (molar masses 16.04 and 44.01 g/mol). It is the CO2 fed in with the biogas, not the exhaust emission.

## License

MIT, see [LICENSE](LICENSE).
