# Backend Type to GUI Node Mapping

One row per unique backend `type` from `scenarios/plant_topology.yaml` (no component IDs).

| Backend Component Type (YAML) | Corresponding GUI Node |
|---|---|
| `type: "CoolingManager"` | `ScenarioComponentNode (fallback)` |
| `type: "ExternalWaterSource"` | `ScenarioComponentNode (fallback)` |
| `type: "WaterPurifier"` | `WaterPurifierNode` |
| `type: "UltraPureWaterTank"` | `UltraPureWaterTankNode` |
| `type: "WaterPumpThermodynamic"` | `ScenarioComponentNode (fallback)` |
| `type: "PowerTransformer"` | `RectifierNode` |
| `type: "SOEC"` | `SOECStackNode` |
| `type: "Interchanger"` | `ScenarioComponentNode (fallback)` |
| `type: "DryCooler"` | `DryCoolerNode` |
| `type: "KnockOutDrum"` | `KnockOutDrumNode` |
| `type: "Chiller"` | `ChillerNode` |
| `type: "HydrogenMultiCyclone"` | `ScenarioComponentNode (fallback)` |
| `type: "ElectricBoiler"` | `ScenarioComponentNode (fallback)` |
| `type: "CompressorSingle"` | `ScenarioComponentNode (fallback)` |
| `type: "DeoxoReactor"` | `DeoxoReactorNode` |
| `type: "Coalescer"` | `CoalescerNode` |
| `type: "PSA Unit"` | `PSAUnitNode` |
| `type: "DrainRecorderMixer"` | `ScenarioComponentNode (fallback)` |
| `type: "StreamSplitter"` | `ScenarioComponentNode (fallback)` |
| `type: "Mixer"` | `MixerNode` |
| `type: "SignalMakeupMixer"` | `ScenarioComponentNode (fallback)` |
| `type: "Attemperator"` | `ScenarioComponentNode (fallback)` |
| `type: "Valve"` | `ValveNode` |
| `type: "SeparationTank"` | `ScenarioComponentNode (fallback)` |
| `type: "PEM"` | `PEMStackNode` |
| `type: "OxygenMakeupNode"` | `ScenarioComponentNode (fallback)` |
| `type: "BiogasSource"` | `ScenarioComponentNode (fallback)` |
| `type: "ProportionalMakeupMixer"` | `ScenarioComponentNode (fallback)` |
| `type: "IntegratedATRPlant"` | `ScenarioComponentNode (fallback)` |
| `type: "ATR_Boiler"` | `ScenarioComponentNode (fallback)` |
| `type: "SyngasPSA"` | `ScenarioComponentNode (fallback)` |
| `type: "DetailedTank"` | `ScenarioComponentNode (fallback)` |
| `type: "DischargeStation"` | `ScenarioComponentNode (fallback)` |

## Validation Summary

- Unique backend component types: `33`
- Types mapped to fallback: `20`
