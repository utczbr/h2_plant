"""
Graph-to-Config adapter: converts visual node graphs into backend configuration
structures used by the simulation context.

Sanitization note:
Only the allowed GUI node classes are supported explicitly. Unknown or removed
classes are treated as generic passive components via fallback behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class FlowType(str, Enum):
    """Port connection types."""

    HYDROGEN = "hydrogen"
    OXYGEN = "oxygen"
    ELECTRICITY = "electricity"
    HEAT = "heat"
    WATER = "water"
    COMPRESSED_H2 = "compressed_h2"
    GAS = "gas"
    STREAM = "stream"
    SIGNAL = "signal"


@dataclass
class Port:
    """Represents an input or output port on a node."""

    name: str
    flow_type: FlowType
    direction: str  # "input" or "output"
    description: str = ""
    unit: str = ""


@dataclass
class GraphNode:
    """Represents a visual node in the editor."""

    id: str
    type: str
    display_name: str
    x: float
    y: float
    properties: Dict[str, Any]
    ports: List[Port]


@dataclass
class GraphEdge:
    """Represents a connection between two nodes."""

    source_node_id: str
    source_port: str
    target_node_id: str
    target_port: str
    flow_type: FlowType

    def validate(self, nodes: Dict[str, GraphNode]) -> None:
        """Ensure both endpoints exist."""
        if self.source_node_id not in nodes:
            raise ValueError(f"Source node {self.source_node_id} not found")
        if self.target_node_id not in nodes:
            raise ValueError(f"Target node {self.target_node_id} not found")


class GraphToConfigAdapter:
    """Main conversion engine for GUI graph -> backend structures."""

    NODE_TYPE_TO_SECTION: Dict[str, Tuple[str, ...]] = {
        # Production
        "PEMStackNode": ("production", "electrolyzer"),
        "SOECStackNode": ("production", "soec"),
        "RectifierNode": ("production", "rectifier"),
        # Thermal
        "ChillerNode": ("thermal_components",),
        "DryCoolerNode": ("thermal_components",),
        "InterchangerNode": ("thermal_components",),
        "ElectricBoilerNode": ("thermal_components",),
        "AttemperatorNode": ("thermal_components",),
        "CoolingManagerNode": ("thermal_components",),
        # Separation
        "PSAUnitNode": ("separation", "psa"),
        "CoalescerNode": ("separation", "coalescer"),
        "KnockOutDrumNode": ("separation", "knockout_drum"),
        "DeoxoReactorNode": ("separation", "deoxo"),
        "HydrogenMultiCycloneNode": ("separation", "cyclone"),
        "SeparationTankNode": ("separation", "separation_tank"),
        "SyngasPSANode": ("separation", "syngas_psa"),
        # Flow control
        "MixerNode": ("flow_control", "mixer"),
        "ValveNode": ("flow_control", "valve"),
        "StreamSplitterNode": ("flow_control", "splitter"),
        "DrainRecorderMixerNode": ("flow_control", "drain_mixer"),
        "SignalMakeupMixerNode": ("flow_control", "signal_makeup_mixer"),
        "ProportionalMakeupMixerNode": ("flow_control", "proportional_makeup_mixer"),
        "OxygenMakeupNode": ("flow_control", "oxygen_makeup"),
        # Water
        "WaterPurifierNode": ("water_treatment", "purifier"),
        "UltraPureWaterTankNode": ("water_treatment", "ultrapure_tank"),
        "ExternalWaterSourceNode": ("water_treatment", "external_source"),
        "WaterPumpThermodynamicNode": ("water_treatment", "pump"),
        # Storage / Delivery
        "DetailedTankNode": ("storage", "detailed_tank"),
        "DischargeStationNode": ("storage", "discharge_station"),
        "CompressorSingleNode": ("storage", "compressor"),
        # Reforming
        "IntegratedATRPlantNode": ("reforming", "atr_plant"),
        "ATRBoilerNode": ("reforming", "atr_boiler"),
        "BiogasSourceNode": ("reforming", "biogas_source"),
    }

    BACKEND_TYPE_TO_NODE_KEY: Dict[str, str] = {
        "PEM": "PEMStackNode",
        "SOEC": "SOECStackNode",
        "PowerTransformer": "RectifierNode",
        "Chiller": "ChillerNode",
        "DryCooler": "DryCoolerNode",
        "Interchanger": "InterchangerNode",
        "ElectricBoiler": "ElectricBoilerNode",
        "Attemperator": "AttemperatorNode",
        "CoolingManager": "CoolingManagerNode",
        "Coalescer": "CoalescerNode",
        "KnockOutDrum": "KnockOutDrumNode",
        "PSA Unit": "PSAUnitNode",
        "DeoxoReactor": "DeoxoReactorNode",
        "HydrogenMultiCyclone": "HydrogenMultiCycloneNode",
        "SeparationTank": "SeparationTankNode",
        "SyngasPSA": "SyngasPSANode",
        "Mixer": "MixerNode",
        "Valve": "ValveNode",
        "StreamSplitter": "StreamSplitterNode",
        "DrainRecorderMixer": "DrainRecorderMixerNode",
        "SignalMakeupMixer": "SignalMakeupMixerNode",
        "ProportionalMakeupMixer": "ProportionalMakeupMixerNode",
        "OxygenMakeupNode": "OxygenMakeupNode",
        "WaterPurifier": "WaterPurifierNode",
        "UltraPureWaterTank": "UltraPureWaterTankNode",
        "ExternalWaterSource": "ExternalWaterSourceNode",
        "WaterPumpThermodynamic": "WaterPumpThermodynamicNode",
        "DetailedTank": "DetailedTankNode",
        "DischargeStation": "DischargeStationNode",
        "CompressorSingle": "CompressorSingleNode",
        "IntegratedATRPlant": "IntegratedATRPlantNode",
        "ATR_Boiler": "ATRBoilerNode",
        "BiogasSource": "BiogasSourceNode",
    }

    PROPERTY_MAPPINGS: Dict[str, Dict[str, str]] = {
        "PEMStackNode": {
            "rated_power_kw": "max_power_mw",
            "efficiency_rated": "base_efficiency",
        },
        "SOECStackNode": {
            "rated_power_kw": "max_power_nominal_mw",
        },
        "RectifierNode": {
            "max_power_kw": "rated_power_mw",
            "conversion_efficiency": "efficiency",
        },
        "ValveNode": {
            "outlet_pressure_bar": "P_out_pa",
            "fluid_type": "fluid",
        },
        "WaterPurifierNode": {
            "output_flow_kgh": "max_flow_kg_h",
        },
        "UltraPureWaterTankNode": {
            "capacity_m3": "volume_m3",
        },
        "ChillerNode": {
            "target_temp_c": "target_temp_k",
        },
    }

    PROPERTY_WHITELISTS: Dict[str, List[str]] = {
        "PEMStackNode": ["rated_power_kw", "efficiency_rated", "component_id"],
        "SOECStackNode": ["rated_power_kw", "operating_temp_c", "component_id"],
        "RectifierNode": ["max_power_kw", "conversion_efficiency", "component_id", "system_group"],
        "ChillerNode": ["cooling_capacity_kw", "target_temp_c", "cop", "pressure_drop_bar", "component_id"],
        "DryCoolerNode": ["fan_power_kw", "approach_temp_c", "pressure_drop_bar", "component_id"],
        "PSAUnitNode": ["recovery_h2", "component_id"],
        "CoalescerNode": ["pressure_drop_bar", "component_id"],
        "KnockOutDrumNode": ["residence_time_s", "component_id"],
        "DeoxoReactorNode": ["component_id"],
        "MixerNode": ["volume_m3", "component_id"],
        "ValveNode": ["outlet_pressure_bar", "fluid_type", "component_id"],
        "WaterPurifierNode": ["output_flow_kgh", "component_id"],
        "UltraPureWaterTankNode": ["capacity_m3", "component_id"],
    }

    _GUI_ONLY_KEYS = {
        "node_color",
        "custom_label",
        "type_",
        "selected",
        "pos",
        "icon",
        "name",
        "disabled",
        "backend_type",
    }

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.errors: List[str] = []

    def add_node(self, node: GraphNode) -> None:
        """Register a visual node."""
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node ID: {node.id}")
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        """Register a connection."""
        edge.validate(self.nodes)
        self.edges.append(edge)

    def to_config_dict(self) -> Dict[str, Any]:
        """Convert the graph to a PlantConfig-style dictionary."""
        config: Dict[str, Any] = {
            "name": self._infer_plant_name(),
            "version": "1.0",
            "production": {},
            "storage": {"source_isolated": False},
            "compression": {},
            "demand": {
                "pattern": "constant",
                "base_demand_kg_h": 50.0,
            },
            "energy_price": {
                "source": "file",
                "price_file": str(Path(__file__).resolve().parent.parent.parent / "data" / "NL_Prices_2024_15min.csv"),
                "wind_data_file": str(Path(__file__).resolve().parent.parent.parent / "data" / "producao_horaria_turbina.csv"),
                "data_resolution_minutes": 15,
            },
            "pathway": {
                "allocation_strategy": "COST_OPTIMAL",
            },
            "simulation": {
                "timestep_hours": 1.0 / 60.0,
                "duration_hours": 8760,
                "checkpoint_interval_hours": 168,
                "max_pow_kwh": 0.0,
            },
            "thermal_components": {
                "chillers": 0,
                "dry_coolers": 0,
                "heat_exchangers": 0,
                "steam_generators": 0,
            },
            "separation": {},
            "flow_control": {},
            "water_treatment": {},
            "control": {},
        }

        for node in self.nodes.values():
            node_key = self._resolve_node_key(node)
            section_info = self.NODE_TYPE_TO_SECTION.get(node_key)
            if section_info is None:
                continue

            node_config = self._extract_node_config(node)
            section = section_info[0]

            if node_key == "ChillerNode":
                config["thermal_components"]["chillers"] += 1
                continue
            if node_key == "DryCoolerNode":
                config["thermal_components"]["dry_coolers"] += 1
                continue

            if len(section_info) == 2:
                subsection = section_info[1]
                if section not in config or not isinstance(config[section], dict):
                    config[section] = {}
                config[section][subsection] = node_config
            else:
                if section not in config or not isinstance(config[section], dict):
                    config[section] = {}
                config[section].update(node_config)

        self._infer_topology_settings(config)
        return config

    def _infer_topology_settings(self, config: Dict[str, Any]) -> None:
        """Set storage topology flags for compatibility with existing consumers."""
        storage_cfg = config.setdefault("storage", {})
        storage_cfg.setdefault("source_isolated", False)

    def _extract_node_config(self, node: GraphNode) -> Dict[str, Any]:
        """Extract backend-ready parameters from node properties."""
        node_key = self._resolve_node_key(node)
        mapping = self.PROPERTY_MAPPINGS.get(node_key, {})

        props = node.properties or {}
        source_props = self._collect_visible_properties(props)

        # Preserve imported scenario params for fallback/legacy nodes.
        scenario_params = props.get("__scenario_params")
        if isinstance(scenario_params, dict):
            merged = dict(scenario_params)
            merged.update(source_props)
            source_props = merged

        whitelist = self.PROPERTY_WHITELISTS.get(node_key)
        if node_key == "ScenarioComponentNode":
            whitelist = None

        config: Dict[str, Any] = {}
        for key, value in source_props.items():
            if whitelist is not None and key not in whitelist:
                continue

            backend_key = mapping.get(key, key)
            coerced = self._coerce_value(value)

            if backend_key in ("max_power_mw", "max_power_nominal_mw") and key == "rated_power_kw":
                coerced = float(coerced) / 1000.0
            elif backend_key == "rated_power_mw" and key == "max_power_kw":
                coerced = float(coerced) / 1000.0
            elif backend_key == "base_efficiency" and isinstance(coerced, (int, float)) and coerced > 1:
                coerced = float(coerced) / 100.0
            elif backend_key in ("efficiency", "isentropic_efficiency", "eta_is", "eta_m"):
                if isinstance(coerced, (int, float)) and coerced > 1:
                    coerced = float(coerced) / 100.0
            elif backend_key == "target_temp_k" and key == "target_temp_c":
                coerced = float(coerced) + 273.15
            elif backend_key == "P_out_pa" and key == "outlet_pressure_bar":
                coerced = float(coerced) * 1e5

            config[backend_key] = coerced

        if node_key in {"PEMStackNode", "SOECStackNode"}:
            config.setdefault("enabled", True)

        return config

    def _collect_visible_properties(self, props: Dict[str, Any]) -> Dict[str, Any]:
        visible: Dict[str, Any] = {}
        for key, value in props.items():
            if key in self._GUI_ONLY_KEYS:
                continue
            if str(key).startswith("__scenario_"):
                continue
            if value is None:
                continue
            visible[key] = value
        return visible

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return value
            try:
                if "." in text:
                    return float(text)
                return int(text)
            except (ValueError, TypeError):
                return value
        return value

    def _infer_plant_name(self) -> str:
        producers = sum(
            1 for node in self.nodes.values() if self._infer_backend_type(node) in {"PEM", "SOEC"}
        )
        return f"Custom Plant ({producers} producers, {len(self.nodes)} nodes)"

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate high-level graph consistency."""
        self.errors = []

        if not self.nodes:
            self.errors.append("Plant graph has no nodes")

        has_production = any(
            self._infer_backend_type(node) in {"PEM", "SOEC"}
            for node in self.nodes.values()
        )
        if not has_production:
            self.errors.append("Plant should contain at least one PEM or SOEC producer")

        for edge in self.edges:
            try:
                edge.validate(self.nodes)
            except ValueError as exc:
                self.errors.append(str(exc))

        return len(self.errors) == 0, self.errors

    def _resolve_backend_ids(self) -> Dict[str, str]:
        """
        Resolve stable backend component IDs for topology emission.

        Priority:
        1) node.properties['component_id'] when non-empty
        2) graph internal node.id
        """
        resolved: Dict[str, str] = {}
        owners: Dict[str, List[str]] = {}

        for graph_node_id, node in self.nodes.items():
            props = node.properties or {}
            raw_component_id = props.get("component_id")
            component_id = str(raw_component_id).strip() if raw_component_id is not None else ""
            backend_id = component_id if component_id else str(node.id)

            resolved[graph_node_id] = backend_id
            owner = f"{node.display_name} ({graph_node_id})"
            owners.setdefault(backend_id, []).append(owner)

        duplicates = {cid: refs for cid, refs in owners.items() if len(refs) > 1}
        if duplicates:
            duplicate_msg = "; ".join(
                f"'{cid}' used by {', '.join(refs)}"
                for cid, refs in sorted(duplicates.items())
            )
            raise ValueError(
                f"Duplicate component IDs detected. Make IDs unique before running: {duplicate_msg}"
            )

        return resolved

    def to_simulation_context(self) -> "SimulationContext":
        """Convert the graph to a validated SimulationContext object."""
        from h2_plant.config.models import (
            ComponentNode,
            EconomicsConfig,
            NodeConnection,
            PEMPhysicsSpec,
            PhysicsConfig,
            SOECPhysicsSpec,
            SimulationConfig,
            SimulationContext,
            TopologyConfig,
        )

        backend_ids = self._resolve_backend_ids()

        topology_nodes: List[ComponentNode] = []
        for node_id, node in self.nodes.items():
            connections: List[NodeConnection] = []
            for edge in self.edges:
                if edge.source_node_id != node_id:
                    continue
                connections.append(
                    NodeConnection(
                        source_port=edge.source_port,
                        target_name=backend_ids.get(edge.target_node_id, edge.target_node_id),
                        target_port=edge.target_port,
                        resource_type=edge.flow_type.value,
                    )
                )

            backend_type = self._infer_backend_type(node)
            topology_nodes.append(
                ComponentNode(
                    id=backend_ids[node_id],
                    type=backend_type,
                    connections=connections,
                    params=self._extract_node_config(node),
                )
            )

        topology = TopologyConfig(nodes=topology_nodes)

        total_pem_mw = 0.0
        total_soec_mw = 0.0
        for node in self.nodes.values():
            backend_type = self._infer_backend_type(node)
            if backend_type == "PEM":
                p_kw = node.properties.get("rated_power_kw", 2500.0)
                total_pem_mw += float(p_kw) / 1000.0
            elif backend_type == "SOEC":
                p_kw = node.properties.get("rated_power_kw", 1000.0)
                total_soec_mw += float(p_kw) / 1000.0

        if total_pem_mw == 0 and total_soec_mw == 0:
            total_pem_mw = 5.0

        physics = PhysicsConfig(
            pem_system=PEMPhysicsSpec(
                max_power_mw=total_pem_mw,
                base_efficiency=0.65,
                kwh_per_kg=56.16,
            ),
            soec_cluster=SOECPhysicsSpec(
                num_modules=6,
                max_power_nominal_mw=total_soec_mw if total_soec_mw > 0 else 2.4,
                optimal_limit=0.80,
            ),
        )

        simulation = SimulationConfig(
            timestep_hours=1.0 / 60.0,
            duration_hours=8760,
            energy_price_file=str(Path(__file__).resolve().parent.parent.parent / "data" / "NL_Prices_2024_15min.csv"),
            wind_data_file=str(Path(__file__).resolve().parent.parent.parent / "data" / "producao_horaria_turbina.csv"),
        )

        economics = self._extract_economics_from_graph()

        return SimulationContext(
            physics=physics,
            topology=topology,
            simulation=simulation,
            economics=economics,
        )

    def _resolve_node_key(self, node: GraphNode) -> str:
        """Resolve a canonical GUI node key, including fallback nodes with backend hints."""
        node_type_short = node.type.split(".")[-1]
        if node_type_short != "ScenarioComponentNode":
            return node_type_short

        props = node.properties or {}
        backend_type = (
            props.get("__scenario_backend_type")
            or props.get("backend_type")
            or ""
        )
        if isinstance(backend_type, str) and backend_type.strip():
            mapped = self.BACKEND_TYPE_TO_NODE_KEY.get(backend_type.strip())
            if mapped:
                return mapped
        return "ScenarioComponentNode"

    def _infer_backend_type(self, node: GraphNode) -> str:
        """Resolve backend type from hidden backend metadata or class mapping."""
        props = node.properties or {}
        backend_type = (
            props.get("__scenario_backend_type")
            or props.get("backend_type")
            or ""
        )
        if isinstance(backend_type, str) and backend_type.strip():
            return backend_type.strip()

        node_key = node.type.split(".")[-1]
        return self._map_node_type(node_key)

    def _map_node_type(self, gui_type: str) -> str:
        """Map GUI node class name (or identifier suffix) to backend component type."""
        mapping = {
            # Allowed explicit GUI nodes
            "PEMStackNode": "PEM",
            "SOECStackNode": "SOEC",
            "RectifierNode": "PowerTransformer",
            "ChillerNode": "Chiller",
            "DryCoolerNode": "DryCooler",
            "CoalescerNode": "Coalescer",
            "KnockOutDrumNode": "KnockOutDrum",
            "PSAUnitNode": "PSA Unit",
            "DeoxoReactorNode": "DeoxoReactor",
            "MixerNode": "Mixer",
            "ValveNode": "Valve",
            "WaterPurifierNode": "WaterPurifier",
            "UltraPureWaterTankNode": "UltraPureWaterTank",
            "ScenarioComponentNode": "PassiveComponent",
            # Identifier suffixes
            "pem": "PEM",
            "soec": "SOEC",
            "rectifier": "PowerTransformer",
            "chiller": "Chiller",
            "dry_cooler": "DryCooler",
            "coalescer": "Coalescer",
            "knockout_drum": "KnockOutDrum",
            "psa": "PSA Unit",
            "psa_unit": "PSA Unit",
            "deoxo": "DeoxoReactor",
            "mixer": "Mixer",
            "valve": "Valve",
            "water_purifier": "WaterPurifier",
            "ultrapure_tank": "UltraPureWaterTank",
        }
        return mapping.get(gui_type, "PassiveComponent")

    def _extract_economics_from_graph(self) -> "EconomicsConfig":
        """Return default economics config for sanitized GUI scope."""
        from h2_plant.config.models import EconomicsConfig

        return EconomicsConfig(
            h2_price_eur_kg=9.60,
            arbitrage_enabled=False,
        )
