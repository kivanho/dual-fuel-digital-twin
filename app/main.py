import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.simulation_service import SimulationService
from app.synthetic_data import CH4_RANGE_PERCENT, DES_RANGE_PERCENT, POWER_RANGE_KW

sim = SimulationService()


@asynccontextmanager
async def lifespan(app):
    """Run the simulation loop in the background while the app is up."""
    async def simulation_loop():
        while True:
            sim.step()
            await asyncio.sleep(sim.dt)

    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()


app = FastAPI(title="Dual-Fuel Engine Digital Twin", lifespan=lifespan)


class SimulationInputs(BaseModel):
    """Operating point. Ranges match the engine map the models are trained on."""
    power_output: float = Field(5, ge=POWER_RANGE_KW[0], le=POWER_RANGE_KW[1],
                                description="Electrical power output [kW]")
    diesel_energy_share: float = Field(10, ge=DES_RANGE_PERCENT[0], le=DES_RANGE_PERCENT[1],
                                       description="Diesel share of the fuel energy [%]")
    methane_percentage: float = Field(85, ge=CH4_RANGE_PERCENT[0], le=CH4_RANGE_PERCENT[1],
                                      description="CH4 volume share of the CH4/CO2 gas mix [%]")


@app.get("/")
def read_root():
    return {"message": "Dual-fuel engine digital twin. See /docs for the API."}


@app.get("/live")
def get_live_snapshot():
    """Current simulation state as a single JSON snapshot."""
    return sim.get_current_state()


@app.get("/live_state")
async def live_state(max_events: Optional[int] = Query(None, ge=1, description="Stop after this many events")):
    """Stream the simulation state as Server-Sent Events, one per step."""
    async def event_stream():
        sent = 0
        while max_events is None or sent < max_events:
            yield f"data: {json.dumps(sim.get_current_state())}\n\n"
            sent += 1
            if max_events is None or sent < max_events:
                await asyncio.sleep(sim.dt)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/update_inputs")
def update_inputs(new_inputs: SimulationInputs):
    """Set a new operating point; the fuel-flow loops move towards it from the current state."""
    try:
        sim.update_inputs(new_inputs.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "Inputs updated."}


@app.post("/reset")
def reset_simulation():
    """Close both fuel valves and reset the controllers."""
    sim.reset()
    return {"status": "Simulation reset."}
