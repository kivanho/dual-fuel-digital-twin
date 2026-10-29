# Dual-Fuel Engine Digital Twin

A digital twin of a diesel/methane dual-fuel generator set (16.5 kVA / 13.2 kWe prime, 18 kVA / 14.4 kWe standby: a 2.2 L four-cylinder diesel at 1500 rpm driving a three-phase 400 V alternator), served as a FastAPI service in Docker.
You set the operating point (electrical power, diesel energy share, methane content of the gas), and the twin predicts the target efficiency and exhaust temperature, works out the fuel mass flows from an energy balance, and lets a PID loop drive the simulated fuel flows towards those targets in real time.

Master's project at IMT Atlantique (Feb–Jun 2025). Team of three; I led the project and did the main implementation.

> **About the data.** The original models were trained on lab measurements from the test bench, which are not public.
> This repository ships a **synthetic engine map** (`app/synthetic_data.py`) with the same columns and physically plausible trends, so everything runs end to end.
> At 100 % CH4 its efficiency and exhaust-temperature curves are anchored to the diesel genset's datasheet (efficiency within 1 percentage point at 50, 75 and 100 % load); the dual-fuel effects on top are illustrative, not measurements of the real engine.

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
 │  PID controller (anti-windup, filtered derivative)                  │
 │          │  actual fuel flows                                       │
 │          ▼                                                          │
 │  Plant outputs ──► power, efficiency, current, torque, CO2 flow     │
 │  Linear regression ──► exhaust temperature (target and actual)      │
 └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
          GET /live (snapshot)   GET /live_state (Server-Sent Events)
```

| File | Role |
|------|------|
| `app/main.py` | FastAPI app, background simulation loop, endpoints |
| `app/simulation_service.py` | Holds models, controller gains and state; one `step()` per tick |
| `app/simulation_core.py` | One simulation step: predict, balance, regulate, derive outputs |
| `app/knnr_model.py` | kNN efficiency model and nearest engine-map point lookup |
| `app/exhaust_temp_model.py` | Linear exhaust-temperature model |
| `app/calculate_massflows.py` | Energy balance from power and efficiency to fuel mass flows |
| `app/regulator.py` | PT1 / PT2 / I / PID regulators (the twin uses PID) |
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

Reset the controller and the simulated flows:

```bash
curl -X POST http://localhost:8000/reset
```

After the PID loop has settled, the state looks like this (values rounded):

```json
{
  "input":  {"power": 12.0, "ch4_percent": 80.0, "des_percent": 20.0},
  "target": {"efficiency": 0.289, "diesel_mass_flow_kg_h": 0.70, "ch4_mass_flow_kg_h": 2.39, "exhaust_temp": 319.8},
  "actual": {"efficiency": 0.289, "power": 11.98, "diesel_mass_flow_kg_h": 0.70, "ch4_mass_flow_kg_h": 2.39,
             "current": 21.6, "torque": 95.3, "exhaust_temp": 319.5, "q_co2_kg_h": 0.60, "ch4_share": 80.0}
}
```

## Run locally

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

## License

MIT, see [LICENSE](LICENSE).
