# Developer Guide: Component Flow Interface

This guide explains how to implement the flow interface for new components in the H2 Plant simulation.

## Overview

All components that participate in resource transfer must implement the `Component` base class interface methods:
1.  `get_ports()`: Define available inputs and outputs.
2.  `receive_input()`: Accept resources from upstream.
3.  `get_output()`: Offer resources to downstream.
4.  `extract_output()`: Finalize transfer and deduct resources.

## The Output Buffering Pattern

To ensure mass conservation and correct flow timing, components should use an **Output Buffer**.
This buffer accumulates production during a timestep (`step()`) and offers it to the `FlowNetwork` in the same timestep.

### Implementation Steps

1.  **Initialize Buffer**: Add a buffer variable in `__init__`.
    ```python
    self._output_buffer_kg = 0.0
    ```

2.  **Reset & Accumulate**: In `step()`, reset the buffer (if not cumulative storage) and add production.
    ```python
    def step(self, t: float) -> None:
        super().step(t)
        # Reset buffer at start of step (for non-storage components)
        # For storage components, you might keep it or handle differently
        self._output_buffer_kg = 0.0
        
        # Calculate production
        produced = ... 
        
        # Add to buffer
        self._output_buffer_kg += produced
    ```

3.  **Offer Output**: In `get_output()`, return the buffered amount as a rate (amount / dt).
    ```python
    def get_output(self, port_name: str) -> Any:
        if port_name == 'out':
            # Return rate based on buffer
            flow_rate = self._output_buffer_kg / self.dt if self.dt > 0 else 0.0
            return Stream(mass_flow_kg_h=flow_rate, ...)
    ```

4.  **Finalize Extraction**: In `extract_output()`, deduct the amount actually taken.
    ```python
    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        if port_name == 'out':
            # Clear buffer or deduct
            self._output_buffer_kg -= amount * self.dt
            # Ensure non-negative
            self._output_buffer_kg = max(0.0, self._output_buffer_kg)
    ```

## Method Details

### `get_ports()`
Returns a dictionary of port metadata.
```python
return {
    'in_port': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
    'out_port': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'}
}
```

### `receive_input(port_name, value, resource_type)`
Called by `FlowNetwork` to push resources into the component.
*   **Return**: The amount actually accepted.
*   **Logic**: Store the input value to be used in the *next* `step()` (or current if order allows, but usually next).

### `get_output(port_name)`
Called by `FlowNetwork` to check available resources.
*   **Return**: A `Stream` object or float value representing the *rate* of flow available.
*   **Important**: Do not modify state here. Just report availability.

### `extract_output(port_name, amount, resource_type)`
Called by `FlowNetwork` *after* a successful transfer to a downstream component.
*   **Logic**: Update internal state (e.g., reduce tank level, clear buffer).
*   **Critical**: This is where you ensure mass is removed from the system so it isn't duplicated.

## Best Practices

*   **Use Streams**: Prefer returning `Stream` objects for fluid flows to carry temperature/pressure data.
*   **Check Resource Types**: Verify `resource_type` in `receive_input` matches expectation.
*   **Handle DT**: Always account for timestep `dt` when converting between rates (kg/h) and amounts (kg).
*   **Mass Balance**: Verify that `Input + Generation = Output + Accumulation`.
