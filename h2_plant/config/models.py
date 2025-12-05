from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union

# --- PHYSICS MODELS ---

class PEMPhysicsSpec(BaseModel):
    max_power_mw: float
    base_efficiency: float
    kwh_per_kg: float
    # Add other specific physics params as needed
    use_polynomials: bool = False
    degradation_rate_per_1000h: float = 0.0 # Example

class SOECPhysicsSpec(BaseModel):
    max_power_nominal_mw: float
    optimal_limit: float
    num_modules: int = 6 # Default standard cluster size
    power_first_step_mw: float = 0.12
    ramp_step_mw: float = 0.24
    degradation_rate_per_1000h: float = 0.0

class PhysicsConfig(BaseModel):
    pem_system: PEMPhysicsSpec
    soec_cluster: SOECPhysicsSpec
    # Global constants can go here
    hhv_h2_kwh_kg: float = 39.4
    lhv_h2_kwh_kg: float = 33.3

# --- TOPOLOGY MODELS ---

class NodeConnection(BaseModel):
    source_port: str
    target_name: str
    target_port: str
    resource_type: str

class ComponentNode(BaseModel):
    id: str
    type: str
    # Optional: override physics for specific instance? 
    # For now, we assume physics is global per type, or we map type to physics spec.
    connections: List[NodeConnection] = []
    params: Dict[str, Any] = {} # Custom parameters for the component instance

class TopologyConfig(BaseModel):
    nodes: List[ComponentNode]

# --- SIMULATION MODELS ---

class SimulationConfig(BaseModel):
    timestep_hours: float
    duration_hours: int
    start_hour: int = 0
    checkpoint_interval_hours: int = 8
    energy_price_file: str
    wind_data_file: str

# --- ECONOMICS MODELS ---

class EconomicsConfig(BaseModel):
    h2_price_eur_kg: float
    ppa_price_eur_mwh: float
    arbitrage_enabled: bool = True

# --- MASTER CONTEXT ---

class SimulationContext(BaseModel):
    physics: PhysicsConfig
    topology: TopologyConfig
    simulation: SimulationConfig
    economics: EconomicsConfig
