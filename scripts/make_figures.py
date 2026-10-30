"""
Draw the README figures (light and dark variants) into docs/.

    python -m app.train                  # the figures use the trained bundle
    python -m scripts.make_figures       # reads MODEL_BUNDLE_DIR or models/synthetic

Left: the kNN efficiency model over the synthetic engine map.
Right: the twin's response to two set-point changes, one step per second.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.simulation_service import SimulationService
from app.synthetic_data import POWER_RANGE_KW, generate_engine_map

DOCS = Path(__file__).resolve().parent.parent / "docs"

THEMES = {
    "light": dict(bg="#FFFFFF", fg="#142E46", muted="#5A6B7B", grid="#E3EBEF",
                  accent="#087E85", second="#4F6F94", third="#9A5B10"),
    "dark": dict(bg="#0D1117", fg="#E6EDF3", muted="#9AA7B2", grid="#21303B",
                 accent="#43BEC4", second="#8DB0D6", third="#E3A55A"),
}


def step_response():
    """Power and fuel flows for 8 → 12 kW at t = 0 and 12 → 6 kW at t = 15 s."""
    sim = SimulationService.from_bundle_dir()
    point = {"diesel_energy_share": 20.0, "methane_percentage": 80.0}
    sim.update_inputs({"power_output": 8.0, **point})
    for _ in range(40):
        sim.step()
    t, power, target, diesel, ch4 = [], [], [], [], []
    for k in range(1, 31):
        p_set = 12.0 if k <= 15 else 6.0
        sim.update_inputs({"power_output": p_set, **point})
        sim.step()
        state = sim.get_current_state()
        t.append(k)
        power.append(state["actual"]["power"])
        target.append(p_set)
        diesel.append(state["actual"]["diesel_mass_flow_kg_h"])
        ch4.append(state["actual"]["ch4_mass_flow_kg_h"])
    return [0] + t, [8.0] + power, [8.0] + target, diesel, ch4


def draw(theme, name):
    c = THEMES[theme]
    plt.rcParams.update({
        "font.size": 10.5, "axes.edgecolor": c["muted"], "axes.labelcolor": c["fg"],
        "xtick.color": c["muted"], "ytick.color": c["muted"], "text.color": c["fg"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": c["grid"], "grid.linewidth": 0.8,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=c["bg"],
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    for ax in (ax1, ax2):
        ax.set_facecolor(c["bg"])
        ax.set_axisbelow(True)

    sim = SimulationService.from_bundle_dir()
    params = sim.bundle.manifest["data"].get("params") or {}
    df = generate_engine_map(params.get("n_points", 400), params.get("seed", 42))
    p = np.linspace(*POWER_RANGE_KW, 120)
    for ch4, color in ((100, c["accent"]), (80, c["second"]), (60, c["third"])):
        near = df[(df["calculated_ch4_share_percent"] - ch4).abs() < 4]
        ax1.scatter(near["power_output"], near["efficiency_electric"], s=9, color=color, alpha=0.35, linewidths=0)
        eff = sim.model.predict(np.column_stack([p, np.full_like(p, ch4)]))
        ax1.plot(p, eff, color=color, lw=2, label=f"{ch4} vol-% CH4")
    ax1.set_xlabel("Electrical power [kW]")
    ax1.set_ylabel("Electrical efficiency [%]")
    ax1.set_title("kNN efficiency model (k = 10, Manhattan)", loc="left", fontsize=11, color=c["fg"])
    ax1.legend(frameon=False, loc="lower right")

    t, power, target, diesel, ch4 = step_response()
    ax2.step(t, target, where="pre", color=c["muted"], lw=1.2, ls="--", label="Set point")
    ax2.plot(t, power, color=c["accent"], lw=2.2, marker="o", ms=3.5, label="Power (actual)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Electrical power [kW]")
    ax2.set_ylim(4, 15.5)
    ax2.set_title("Response to set-point changes (PI flow loops)", loc="left", fontsize=11, color=c["fg"])
    flows = ax2.twinx()
    flows.plot(t[1:], ch4, color=c["second"], lw=1.4, label="CH4 flow")
    flows.plot(t[1:], diesel, color=c["third"], lw=1.4, label="Diesel flow")
    flows.set_ylabel("Fuel flow [kg/h]")
    flows.set_ylim(0, 4.65)
    flows.grid(False)
    flows.spines["right"].set_visible(True)
    flows.spines["top"].set_visible(False)
    handles = ax2.get_legend_handles_labels()[0] + flows.get_legend_handles_labels()[0]
    ax2.legend(handles, [h.get_label() for h in handles], frameon=False, loc="upper center", ncol=4, fontsize=9, handlelength=1.6, columnspacing=1.0)

    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    fig.savefig(DOCS / f"{name}_{theme}.png", dpi=150, facecolor=c["bg"])
    plt.close(fig)


def main():
    for theme in THEMES:
        draw(theme, "twin_overview")
    print(f"Wrote figures to {DOCS}")


if __name__ == "__main__":
    main()
