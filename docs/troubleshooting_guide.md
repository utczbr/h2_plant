# Troubleshooting Guide

This guide helps diagnose and resolve common issues encountered when configuring and running the H2 Plant simulation.

## Configuration Errors

### "Topology references unknown component"
**Error**: `ConfigurationError: Topology references unknown source component: X`
**Cause**: The `source` or `target` ID in your `topology` section does not match any `component_id` defined in the component sections (e.g., `pem_system`, `storage`).
**Fix**: Check for typos in `h2plant.yaml`. Ensure the component is actually defined in the relevant system section.

### "Port is not an output"
**Error**: `ConfigurationError: Port X on Y is not an output`
**Cause**: You are trying to use an input port as a source in the topology.
**Fix**: Check the component's documentation or code (`get_ports()`) to see which ports are outputs. For example, tanks usually have `h2_out` (output) and `h2_in` (input).

### "Duplicate Component ID"
**Error**: `DuplicateComponentError: Component ID 'X' already registered`
**Cause**: Two components have the same `component_id`.
**Fix**: Ensure all `component_id` fields in `h2plant.yaml` are unique across the entire plant.

## Simulation Errors

### Mass Balance Error
**Symptom**: Simulation finishes but reports a Mass Balance Error > 0.01%.
**Cause**:
1.  **Unconnected Outputs**: An output port is producing mass but it's not connected to anything (or connected to a sink that doesn't track it).
2.  **Component Logic**: A component might be generating mass without accounting for it, or consuming it without deducting it.
3.  **Execution Order**: (Developer) Ensure `step_all` runs before `execute_flows` to capture production in the same timestep.

**Debugging**:
*   Run `tests/debug_mass_balance.py` to see a step-by-step log of mass transfer.
*   Check if the error equals exactly one hour of production (implies timing/lag issue).
*   Check if the error accumulates over time (implies leak or duplication).

### "Unknown output type"
**Warning**: `Unknown output type from component:port: <class 'NoneType'>`
**Cause**: A component's `get_output()` method returned `None` instead of a float or `Stream`.
**Fix**: Ensure `get_output()` always returns a valid value (e.g., 0.0) even if the component is off.

## Performance Issues

### Slow Simulation
**Cause**:
1.  **Too many components**: Each connection adds overhead.
2.  **Complex Models**: Components like `ATRReactor` using complex interpolation might be slow if not optimized.
3.  **Logging**: High log levels (DEBUG) write a lot of text to disk/console.

**Fix**:
*   Increase timestep `dt` (e.g., 1h -> 4h) if acceptable accuracy.
*   Disable detailed logging in production runs.
*   Use Numba-optimized components where possible.
