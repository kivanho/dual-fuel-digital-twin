# Improved regulator.py

def apply_regulation(sim_state, sim_vel, integral, previous_error, targets, dt,
                     controller_type, T, zeta, omega_0, kp, ki, kd,
                     output_limits=(-float('inf'), float('inf')), derivative_filter_alpha=0.1):

    """
    Apply per-variable control with improved PID (anti-windup, derivative filter, output clamp)

    output_limits: tuple (min_output, max_output)
    derivative_filter_alpha: float (0-1), lower = more smoothing
    """

    # Store last derivative for filtering
    if "derivative_filtered" not in sim_state:
        sim_state["derivative_filtered"] = {var: 0.0 for var in controller_type}

    for var, target in targets.items():
        method = controller_type.get(var, "pt1")  # default to PT1

        if method == "pt1":
            sim_state[var] += (target - sim_state[var]) * dt / T.get(var, 1.0)

        elif method == "pt2":
            a = 2 * zeta.get(var, 0.5) * omega_0.get(var, 1.0)
            b = omega_0.get(var, 1.0) ** 2
            sim_vel[var] += dt * (b * (target - sim_state[var]) - a * sim_vel[var])
            sim_state[var] += dt * sim_vel[var]

        elif method == "i":
            integral[var] += (target - sim_state[var]) * dt
            sim_state[var] += ki.get(var, 0.1) * integral[var] * dt

        elif method == "pid":
            error = target - sim_state[var]

            # Integral with Anti-Windup
            integral[var] += error * dt
            integral[var] = max(min(integral[var], 1000.0), -1000.0)  # Example clamping

            # Derivative calculation + Filter
            raw_derivative = (error - previous_error.get(var, 0.0)) / dt
            filtered_derivative = (
                derivative_filter_alpha * raw_derivative +
                (1 - derivative_filter_alpha) * sim_state["derivative_filtered"][var]
            )
            sim_state["derivative_filtered"][var] = filtered_derivative

            # PID calculation
            control_output = (
                kp.get(var, 1.0) * error +
                ki.get(var, 0.1) * integral[var] +
                kd.get(var, 0.01) * filtered_derivative
            )

            # Output Clamping
            control_output = max(min(control_output, output_limits[1]), output_limits[0])

            # Update state
            sim_state[var] += control_output

            # Store previous error
            previous_error[var] = error

    return sim_state, sim_vel, integral, previous_error