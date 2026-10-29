"""
Fuel-flow control loop: a PI controller driving a first-order valve model.

Each fuel line (diesel, CH4) is modelled as a valve whose flow follows the
commanded flow with a first-order lag (PT1, time constant `valve_time_constant`).
The PI controller sets that command from the flow error. Its gains cancel the
valve pole (ki = kp / T), so the closed loop behaves like a first-order system
with time constant T / kp = 0.8 s: no overshoot, and within 2 % of a new
target after about 3 s with the default gains. The command is clamped to the
valve's physical range, and the integrator stops while the command is
saturated (anti-windup).
"""
from dataclasses import dataclass, field


@dataclass
class FlowLoop:
    kp: float = 2.5
    ki: float = 1.25                   # [1/s]
    valve_time_constant: float = 2.0   # [s]
    min_flow: float = 0.0              # [kg/h]
    max_flow: float = float("inf")     # [kg/h]
    flow: float = 0.0                  # actual flow through the valve [kg/h]
    integral: float = field(default=0.0, repr=False)

    def step(self, target, dt):
        """Advance the loop by dt seconds towards `target` and return the actual flow."""
        error = target - self.flow
        command = self.kp * error + self.ki * self.integral
        saturated = command > self.max_flow or command < self.min_flow
        command = min(max(command, self.min_flow), self.max_flow)

        # Anti-windup: only integrate while the command is not clamped,
        # or when the error would drive it back out of saturation
        if not saturated or (command >= self.max_flow and error < 0) or (command <= self.min_flow and error > 0):
            self.integral += error * dt

        self.flow += (command - self.flow) * dt / self.valve_time_constant
        return self.flow

    def reset(self):
        self.flow = 0.0
        self.integral = 0.0
