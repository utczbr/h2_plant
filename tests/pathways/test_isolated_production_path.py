import pytest
from h2_plant.pathways.isolated_production_path import IsolatedProductionPath
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component import Component

# Mock components for testing
class MockSource(Component):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.power_input_mw = 0.0
        self.h2_output_kg = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_energy_kwh = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)
        self.h2_output_kg = self.power_input_mw * 10 # Simplified production
        self.cumulative_cost += self.h2_output_kg * 2.0 # $2/kg
        self.cumulative_energy_kwh += self.power_input_mw * 1000 * self.dt

    def get_state(self) -> dict:
        return {**super().get_state(), 'h2_output_kg': self.h2_output_kg}

class MockStorage(Component):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mass = 0.0
        self.capacity = 1000.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)

    def get_state(self) -> dict:
        return {**super().get_state(), 'mass': self.mass}
    
    def fill(self, mass: float) -> tuple[float, float]:
        can_store = self.capacity - self.mass
        stored = min(mass, can_store)
        self.mass += stored
        overflow = mass - stored
        return stored, overflow
    
    def discharge(self, mass: float) -> float:
        discharged = min(mass, self.mass)
        self.mass -= discharged
        return discharged

    def get_total_mass(self) -> float:
        return self.mass

    def get_available_capacity(self) -> float:
        return self.capacity - self.mass

class MockCompressor(Component):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transfer_mass_kg = 0.0
        self.actual_mass_transferred_kg = 0.0
        self.energy_consumed_kwh = 0.0
        self.max_flow_kg_h = 100.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)
        self.actual_mass_transferred_kg = min(self.transfer_mass_kg, self.max_flow_kg_h * self.dt)
        self.energy_consumed_kwh = self.actual_mass_transferred_kg * 0.5 # 0.5 kWh/kg


    def get_state(self) -> dict:
        return {**super().get_state()}

@pytest.fixture
def mock_components():
    """Fixture to provide a registry with mock components for a pathway."""
    registry = ComponentRegistry()
    registry.register('source', MockSource())
    registry.register('lp_storage', MockStorage())
    registry.register('hp_storage', MockStorage())
    registry.register('compressor', MockCompressor())
    registry.initialize_all(dt=1.0)
    return registry

def test_pathway_initialization(mock_components):
    """Test pathway initializes and resolves component references."""
    path = IsolatedProductionPath(
        pathway_id='test_path',
        source_id='source',
        lp_storage_id='lp_storage',
        hp_storage_id='hp_storage',
        compressor_id='compressor'
    )
    
    path.initialize(dt=1.0, registry=mock_components)
    
    assert path._initialized
    assert path._source is not None
    assert path._lp_storage is not None
    assert path._hp_storage is not None
    assert path._compressor is not None

def test_pathway_production_flow(mock_components):
    """Test complete production flow through pathway."""
    path = IsolatedProductionPath(
        pathway_id='test_path',
        source_id='source',
        lp_storage_id='lp_storage',
        hp_storage_id='hp_storage',
        compressor_id='compressor'
    )
    path.initialize(dt=1.0, registry=mock_components)
    
    # Set production target
    path.production_target_kg_h = 50.0 
    
    # Execute timestep
    path.step(0.0)
    
    # Verify production occurred and was stored in LP storage
    # Expected production = 50.0 (target_kg_h) * 0.077 (pathway factor) * 10 (mock factor) = 38.5
    assert abs(path.h2_produced_kg - 38.5) < 1e-6
    assert abs(path.h2_stored_lp_kg - 38.5) < 1e-6
    assert path.h2_stored_hp_kg == 0.0

def test_pathway_compression_flow(mock_components):
    """Test LP to HP transfer logic."""
    path = IsolatedProductionPath(
        pathway_id='test_path',
        source_id='source',
        lp_storage_id='lp_storage',
        hp_storage_id='hp_storage',
        compressor_id='compressor',
        lp_to_hp_threshold_kg=100.0 # Lower threshold for easier testing
    )
    path.initialize(dt=1.0, registry=mock_components)

    # Step 1: Produce enough to trigger compression in the next step.
    # Target 150kg production -> target_kg_h = 150 / (0.077 * 10) = 194.8...
    path.production_target_kg_h = 150.0 / (0.077 * 10)
    path.step(0.0)

    # After this step, LP storage should have 150kg, and compression should have run.
    # LP tank has 150kg, we transfer min(150, 100*1) = 100kg
    # LP tank should have 50kg left, HP tank should have 100kg.
    assert abs(path.h2_stored_lp_kg - 50.0) < 1e-6
    assert abs(path.h2_stored_hp_kg - 100.0) < 1e-6
