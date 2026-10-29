from fastapi import FastAPI, HTTPException, Query
import asyncio
import json
from typing import Optional
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.engine_specs import RATED_POWER_KW
from app.simulation_service import SimulationService

# Initialize FastAPI app and simulation service
app = FastAPI(title="Dual-Fuel Engine Digital Twin")
sim = SimulationService()
_loop_task = None

@app.on_event("startup")
async def start_background_task():
    """Start the simulation loop as a background task."""
    global _loop_task

    async def simulation_loop():
        while True:
            sim.step()  # Advance the simulation by one step
            await asyncio.sleep(sim.dt)  # Wait for the simulation time step
    _loop_task = asyncio.create_task(simulation_loop())

@app.on_event("shutdown")
async def stop_background_task():
    """Stop the simulation loop."""
    if _loop_task is not None:
        _loop_task.cancel()

@app.get("/")
def read_root():
    """Welcome message for the API."""
    return {"message": "Welcome to the Digital Twin Simulation API"}

@app.get("/live")
def get_live_snapshot():
    """
    Return the current simulation state as a standard GET response.
    This is not a stream — it returns a fresh snapshot on each request.
    """
    try:
        return sim.get_current_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/live_state")
async def live_state(max_events: Optional[int] = Query(None, ge=1, description="Stop after this many events")):
    """Stream live simulation state updates as Server-Sent Events."""
    async def event_stream():
        sent = 0
        while max_events is None or sent < max_events:
            try:
                yield f"data: {json.dumps(sim.get_current_state())}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
            sent += 1
            if max_events is None or sent < max_events:
                await asyncio.sleep(sim.dt)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Define a Pydantic model for input validation
class SimulationInputs(BaseModel):
    power_output: float = Field(5, ge=0.01, le=RATED_POWER_KW, description="Desired electrical power output in kW")
    diesel_energy_share: float = Field(10, ge=0, le=100, description="Diesel energy share (0 to 100)")
    methane_percentage: float = Field(85, gt=0, le=100, description="Methane share of the CH4/CO2 gas mix (0 to 100)")

@app.post("/update_inputs")
def update_inputs(new_inputs: SimulationInputs):
    """Update the input parameters for the simulation."""
    try:
        # Convert Pydantic model to dictionary and pass to the simulation
        sim.update_inputs(new_inputs.dict())
        return {"status": "Inputs updated successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
def reset_simulation():
    """Reset the simulation to its initial state."""
    try:
        sim.reset()
        return {"status": "Simulation reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
