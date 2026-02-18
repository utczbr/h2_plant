"""
Generic scenario component node used as a safe visual fallback for backend types
without dedicated GUI node implementations.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from h2_plant.gui.nodes.base_node import ConfigurableNode


class ScenarioComponentNode(ConfigurableNode):
    """
    Read-only oriented generic node for scenario imports.

    It supports dynamic runtime port creation so imported topology port names are
    preserved exactly.
    """

    __identifier__ = "nodes.Scenario"
    NODE_NAME = "Scenario Component"

    def __init__(self):
        self._syncing_ports = False
        self._param_property_keys: set = set()  # tracks surfaced param properties
        super(ScenarioComponentNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self) -> None:
        # Ports are configured dynamically from scenario metadata.
        return None

    def _init_properties(self) -> None:
        self.add_text_property("component_id", default="Scenario_Node", tab="Scenario")
        self.add_text_property("backend_type", default="PassiveComponent", tab="Scenario")
        self.create_property("__scenario_component_id", value="", widget_type=0)
        self.create_property("__scenario_backend_type", value="", widget_type=0)
        self.create_property("__scenario_inputs", value=[], widget_type=0)
        self.create_property("__scenario_outputs", value=[], widget_type=0)
        self.create_property("__scenario_params", value={}, widget_type=0)
        self.add_color_property("node_color", default=(140, 160, 180), tab="Custom")
        self.add_spacer("collapse_spacer", height=60)

    def set_property(self, name, value, push_undo=True):
        """Rebuild dynamic ports when persistence restores scenario port metadata.
        Sync edits to surfaced parameter properties back to __scenario_params."""
        super(ScenarioComponentNode, self).set_property(name, value, push_undo=push_undo)
        if name in {"__scenario_inputs", "__scenario_outputs"} and not self._syncing_ports:
            self._restore_ports_from_properties()
        # Sync edits from visible parameter properties → hidden __scenario_params dict
        if name in self._param_property_keys:
            current_params = dict(self.get_property("__scenario_params") or {})
            # Parse with YAML scalar rules to preserve numeric/bool/list types
            import yaml
            try:
                parsed = yaml.safe_load(value) if isinstance(value, str) else value
            except Exception:
                parsed = value
            current_params[name] = parsed
            super(ScenarioComponentNode, self).set_property(
                "__scenario_params", current_params, push_undo=False
            )
        # Re-surface params when __scenario_params is restored from persistence
        if name == "__scenario_params" and isinstance(value, dict) and value:
            self._restore_surfaced_params()

    def configure_from_scenario(
        self,
        component_id: str,
        backend_type: str,
        input_ports: Iterable[str],
        output_ports: Iterable[str],
        params: Dict[str, Any] | None = None,
    ) -> None:
        """Apply scenario identity and dynamic ports."""
        super(ScenarioComponentNode, self).set_property("component_id", str(component_id))
        super(ScenarioComponentNode, self).set_property("backend_type", str(backend_type))
        super(ScenarioComponentNode, self).set_property("__scenario_component_id", str(component_id))
        super(ScenarioComponentNode, self).set_property("__scenario_backend_type", str(backend_type))
        if params is not None:
            super(ScenarioComponentNode, self).set_property("__scenario_params", dict(params))
            self._surface_params_as_properties(params)
        self.configure_scenario_ports(input_ports, output_ports)

    def configure_scenario_ports(
        self,
        input_ports: Iterable[str],
        output_ports: Iterable[str],
    ) -> None:
        """Ensure all listed ports exist on the node."""
        inputs = self._coerce_port_list(input_ports)
        outputs = self._coerce_port_list(output_ports)

        existing_inputs = {port.name() for port in self.input_ports()}
        existing_outputs = {port.name() for port in self.output_ports()}

        for port_name in inputs:
            if port_name not in existing_inputs:
                self.add_input(
                    port_name,
                    flow_type=self._infer_flow_type(port_name),
                    multi_input=True,
                )
                existing_inputs.add(port_name)

        for port_name in outputs:
            if port_name not in existing_outputs:
                self.add_output(
                    port_name,
                    flow_type=self._infer_flow_type(port_name),
                    multi_output=True,
                )
                existing_outputs.add(port_name)

        self._syncing_ports = True
        try:
            super(ScenarioComponentNode, self).set_property("__scenario_inputs", inputs)
            super(ScenarioComponentNode, self).set_property("__scenario_outputs", outputs)
        finally:
            self._syncing_ports = False

    def _restore_ports_from_properties(self) -> None:
        inputs = self._coerce_port_list(self.get_property("__scenario_inputs"))
        outputs = self._coerce_port_list(self.get_property("__scenario_outputs"))
        self.configure_scenario_ports(inputs, outputs)

    def _surface_params_as_properties(self, params: Dict[str, Any]) -> None:
        """Create visible, editable properties in a 'Parameters' tab for each param."""
        for key, value in (params or {}).items():
            prop_name = str(key)
            if prop_name in self.properties():
                # Already registered — just update value
                super(ScenarioComponentNode, self).set_property(prop_name, str(value))
            else:
                self.add_text_property(prop_name, default=str(value), tab="Parameters")
            self._param_property_keys.add(prop_name)

    def _restore_surfaced_params(self) -> None:
        """Re-surface params from __scenario_params after property restoration."""
        params = self.get_property("__scenario_params")
        if isinstance(params, dict) and params:
            self._surface_params_as_properties(params)

    @staticmethod
    def _coerce_port_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            ports = [str(item).strip() for item in value]
        else:
            ports = [token.strip() for token in str(value).split(",")]
        return [port for port in ports if port]

    @staticmethod
    def _infer_flow_type(port_name: str) -> str:
        name = str(port_name).lower()
        if any(token in name for token in ("signal", "control", "demand")):
            return "signal"
        if any(token in name for token in ("water", "steam", "drain", "makeup", "ultrapure")):
            return "water"
        if any(token in name for token in ("o2", "oxygen")):
            return "oxygen"
        if any(token in name for token in ("power", "electric", "grid")):
            return "electricity"
        if any(token in name for token in ("h2", "hydrogen", "purified", "compressed")):
            return "hydrogen"
        if any(token in name for token in ("gas", "inlet", "feed", "syngas", "tail")):
            return "gas"
        if any(token in name for token in ("heat", "thermal", "cooling", "duty")):
            return "heat"
        return "stream"
