# Plant Topology Configuration Guide

This guide explains how to configure the H2 Plant simulation topology using the `h2plant.yaml` file.

## Overview

The plant topology defines how components are connected and how resources (water, hydrogen, electricity, etc.) flow between them. The configuration uses a YAML format that is both human-readable and machine-parsable.

## YAML Structure

The `h2plant.yaml` file is divided into several sections:

1.  **Plant Metadata**: Basic information about the plant.
2.  **Simulation Settings**: Time steps, duration, etc.
3.  **Component Definitions**: Detailed configuration for each subsystem (PEM, SOEC, ATR, Storage, etc.).
4.  **Topology**: A list of connections between components.

### Topology Section

The `topology` section is a list of connection objects. Each connection specifies a source, a target, and the resource being transferred.

```yaml
topology:
  - source: "electrolyzer_1"
    port: "h2_out"
    target: "lp_tank_1"
    target_port: "h2_in"
    resource: "hydrogen"
```

## Connection Fields

| Field | Description | Example |
| :--- | :--- | :--- |
| `source` | The `component_id` of the source component. | `pem_stack_1` |
| `port` | The output port name on the source component. | `h2_out` |
| `target` | The `component_id` of the target component. | `compressor_1` |
| `target_port` | The input port name on the target component. | `h2_in` |
| `resource` | The type of resource being transferred. | `hydrogen`, `water`, `electricity` |

## Port Naming Conventions

Components follow a standard naming convention for ports:

*   **Inputs**: Suffix `_in` (e.g., `water_in`, `electricity_in`)
*   **Outputs**: Suffix `_out` (e.g., `h2_out`, `heat_out`)

### Common Ports by Component Type

**Electrolyzers (PEM/SOEC)**
*   Inputs: `electricity_in`, `water_in` (or `steam_in`)
*   Outputs: `h2_out`, `o2_out`, `heat_out`

**Compressors**
*   Inputs: `h2_in`, `electricity_in`
*   Outputs: `h2_out`

**Tanks**
*   Inputs: `h2_in`
*   Outputs: `h2_out`

**ATR Reactors**
*   Inputs: `biogas_in`, `steam_in`, `o2_in`
*   Outputs: `h2_out`, `offgas_out`, `heat_out`

## Example Configuration

Here is an example of connecting an Electrolyzer to a Low-Pressure Tank, then to a Compressor, and finally to a High-Pressure Tank.

```yaml
topology:
  # Electrolyzer -> LP Tank
  - source: "pem_stack_1"
    port: "h2_out"
    target: "lp_tank_1"
    target_port: "h2_in"
    resource: "hydrogen"

  # LP Tank -> Compressor
  - source: "lp_tank_1"
    port: "h2_out"
    target: "compressor_1"
    target_port: "h2_in"
    resource: "hydrogen"

  # Compressor -> HP Tank
  - source: "compressor_1"
    port: "h2_out"
    target: "hp_tank_1"
    target_port: "h2_in"
    resource: "hydrogen"
```

## Validation

The system validates the topology on startup:
1.  **Existence**: Checks if all source and target IDs exist.
2.  **Port Validity**: Checks if the specified ports exist on the components.
3.  **Direction**: Ensures connections go from Output -> Input.
4.  **Resource Match**: Warns if the connection resource type doesn't match the port definitions.

## Troubleshooting

*   **"Topology references unknown component"**: Check your `component_id` in the component definition sections vs the topology.
*   **"Port is not an output"**: You might be trying to connect to an input port as a source.
*   **Mass Balance Errors**: Ensure you haven't created a loop where mass is duplicated, or left an output unconnected if it's critical for balance (though unconnected outputs usually just mean lost resource).
