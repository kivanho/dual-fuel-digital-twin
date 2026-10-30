"""
FastAPI service around one simulated dual-fuel engine.

Startup loads a trained model bundle (MODEL_BUNDLE_DIR, default models/synthetic;
train one with `python -m app.train`) and creates the single shared
SimulationService; a background task steps it once per simulated second.
Run with ONE worker process (`uvicorn app.main:app`): see
app/simulation_service.py for the ownership model.

Environment
  MODEL_BUNDLE_DIR            bundle directory (default models/synthetic)
  UNSUPPORTED_POINT_POLICY    "reject" (default, HTTP 422) or "warn" (accept, report in `warnings`)
  SIM_BACKGROUND_LOOP         "1" (default) steps the simulation in the background; "0" only on demand
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager, suppress
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.bundle import BundleError, load_bundle
from app.engine_specs import RATED_POWER_KW
from app.simulation_service import POLICIES, SimulationService, UnsupportedOperatingPoint

SUPPORTED_WORKERS = 1


class SimulationInputs(BaseModel):
    """
    Requested operating point. The bounds here are the genset's equipment limits;
    the trained bundle's supported domain (training range and support distance)
    is checked on top in the endpoint. See GET /model for the current domain.
    """
    power_output: float = Field(5, gt=0, le=RATED_POWER_KW, description="Electrical power output [kW]")
    diesel_energy_share: float = Field(10, ge=0, le=100, description="Diesel share of the fuel energy [%]")
    methane_percentage: float = Field(85, gt=0, le=100, description="CH4 volume share of the CH4/CO2 gas mix [%]")


def create_app(bundle_dir: Optional[str] = None, unsupported_policy: Optional[str] = None,
               background_loop: Optional[bool] = None) -> FastAPI:
    """Build the app. Arguments override the environment variables documented above."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        policy = unsupported_policy or os.environ.get("UNSUPPORTED_POINT_POLICY", "reject")
        if policy not in POLICIES:
            raise RuntimeError(f"UNSUPPORTED_POINT_POLICY must be one of {POLICIES}, got {policy!r}")
        try:
            bundle = load_bundle(bundle_dir)
        except BundleError as e:
            raise RuntimeError(f"cannot start: {e}") from e
        sim = SimulationService(bundle, unsupported_policy=policy)
        app.state.sim = sim
        app.state.ready = True

        run_loop = background_loop if background_loop is not None else os.environ.get("SIM_BACKGROUND_LOOP", "1") != "0"

        async def simulation_loop():
            while True:
                await asyncio.to_thread(sim.step)
                await asyncio.sleep(sim.dt)

        task = asyncio.create_task(simulation_loop()) if run_loop else None
        try:
            yield
        finally:
            app.state.ready = False
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="Dual-Fuel Engine Digital Twin", lifespan=lifespan,
                  description="Simulation and inference-service prototype: one simulated engine shared by all clients.")
    app.state.ready = False
    app.state.sim = None

    def get_sim(request: Request) -> SimulationService:
        sim = request.app.state.sim
        if sim is None or not request.app.state.ready:
            raise HTTPException(status_code=503, detail="simulation not ready")
        return sim

    @app.get("/")
    def read_root():
        return {"message": "Dual-fuel engine digital twin (simulation prototype). See /docs for the API."}

    @app.get("/health")
    def health(request: Request):
        """Liveness plus what is loaded. Always 200 once the process answers."""
        sim = request.app.state.sim
        return {
            "status": "ok",
            "ready": bool(request.app.state.ready and sim is not None),
            "model_version": sim.model_version if sim else None,
            "bundle_path": sim.bundle.path if sim else None,
            "simulation": sim.status if sim else None,
            "time_s": sim.time if sim else None,
            "unsupported_point_policy": sim.unsupported_policy if sim else None,
            "supported_workers": SUPPORTED_WORKERS,
        }

    @app.get("/ready")
    def ready(request: Request):
        """200 when the bundle is loaded and the simulator exists, otherwise 503."""
        sim = request.app.state.sim
        if sim is None or not request.app.state.ready:
            return JSONResponse(status_code=503, content={"ready": False})
        return {"ready": True, "model_version": sim.model_version}

    @app.get("/model")
    def model_info(sim: SimulationService = Depends(get_sim)):
        """Manifest of the loaded bundle: features, units, supported domain, config, data fingerprint, evaluation."""
        return sim.bundle.manifest

    @app.get("/live")
    def get_live_snapshot(sim: SimulationService = Depends(get_sim)):
        """Current simulation state as a single consistent JSON snapshot."""
        return sim.get_current_state()

    @app.get("/live_state")
    async def live_state(max_events: Optional[int] = Query(None, ge=1, description="Stop after this many events"),
                         sim: SimulationService = Depends(get_sim)):
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
    def update_inputs(new_inputs: SimulationInputs, sim: SimulationService = Depends(get_sim)):
        """
        Set a new operating point; the fuel-flow loops move towards it from the current state.
        Unsupported points are rejected with 422 (policy "reject") or accepted with `warnings` (policy "warn").
        """
        try:
            check = sim.update_inputs(new_inputs.model_dump())
        except UnsupportedOperatingPoint as e:
            raise HTTPException(status_code=422, detail={"message": "operating point outside the supported domain",
                                                         "violations": e.check.violations,
                                                         "support": e.check.to_dict()})
        return {"status": "Inputs updated.", "input_support": check.to_dict(),
                "warnings": [] if check.supported else check.violations}

    @app.post("/reset")
    def reset_simulation(sim: SimulationService = Depends(get_sim)):
        """Restart the simulation at its initial condition: t = 0, valves closed, default operating point, running."""
        sim.reset()
        return {"status": "Simulation reset.", "state": sim.get_current_state()}

    @app.post("/stop")
    def stop_simulation(sim: SimulationService = Depends(get_sim)):
        """Close both fuel valves; the engine winds down and stays stopped until /start."""
        sim.stop()
        return {"status": "Simulation stopped.", "simulation": sim.status}

    @app.post("/start")
    def start_simulation(sim: SimulationService = Depends(get_sim)):
        """Resume towards the requested operating point after /stop."""
        sim.start()
        return {"status": "Simulation started.", "simulation": sim.status}

    return app


app = create_app()
