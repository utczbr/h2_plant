"""
Composite component for systems with hierarchical subsystems.

Extends the basic Component ABC to support nested component architectures.
"""

from typing import Dict, Any, List
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class CompositeComponent(Component):
    """
    Base class for components containing subsystems.

    Automatically handles initialization and stepping of child components.
    """

    def __init__(self):
        super().__init__()
        self._subsystems: List[Component] = []

    def add_subsystem(self, name: str, component: Component) -> None:
        """Register a subsystem."""
        self._subsystems.append(component)
        setattr(self, name, component)

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        for subsystem in self._subsystems:
            subsystem.initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)
        # Override in subclass to define subsystem interactions
        for subsystem in self._subsystems:
            subsystem.step(t)

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'subsystems': {
                getattr(subsystem, 'component_id', f'subsystem_{i}'): subsystem.get_state()
                for i, subsystem in enumerate(self._subsystems)
            }
        }