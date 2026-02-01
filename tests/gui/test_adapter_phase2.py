"""
Test Phase 2 GUI Components (Full Node Set).
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode, GraphEdge, FlowType
from h2_plant.config.plant_builder import PlantBuilder

def test_adapter_full_plant():
    print("Testing GraphToConfigAdapter with FULL plant nodes...")
    adapter = GraphToConfigAdapter()
    
    # 1. Production: Electrolyzer & ATR
    elec = GraphNode("elec_1", "ElectrolyzerNode", "Electrolyzer", 0, 0, 
                     {"max_power_mw": 10.0, "base_efficiency": 0.7}, [])
    atr = GraphNode("atr_1", "ATRSourceNode", "ATR", 0, 100, 
                    {"max_ng_flow_kg_h": 200.0, "efficiency": 0.8}, [])
    
    # 2. Storage: LP & HP Tanks
    lp_tank = GraphNode("lp_1", "LPTankNode", "LP Tanks", 200, 0, 
                        {"count": 5, "capacity_kg": 60.0, "pressure_bar": 30.0}, [])
    hp_tank = GraphNode("hp_1", "HPTankNode", "HP Tanks", 400, 0, 
                        {"count": 10, "capacity_kg": 250.0, "pressure_bar": 350.0}, [])
    
    # 3. Compression
    fill_comp = GraphNode("comp_1", "FillingCompressorNode", "Filling Comp", 300, 0, 
                          {"max_flow_kg_h": 150.0, "inlet_pressure_bar": 30.0, "outlet_pressure_bar": 350.0, "efficiency": 0.75}, [])
    
    # 4. Utilities
    battery = GraphNode("bat_1", "BatteryNode", "Battery", -100, 0, 
                        {"capacity_kwh": 2000.0}, [])
    water = GraphNode("water_1", "WaterTreatmentNode", "Water Plant", -100, 100, 
                      {"max_flow_m3h": 15.0}, [])
    
    # 5. Logic
    demand = GraphNode("dem_1", "DemandSchedulerNode", "Demand", 500, 0, 
                       {"base_demand_kg_h": 80.0}, [])
    
    # Add all nodes
    for n in [elec, atr, lp_tank, hp_tank, fill_comp, battery, water, demand]:
        adapter.add_node(n)
    
    # Create connections (Topology inference test)
    # Elec -> LP Tank (Shared storage scenario)
    adapter.add_edge(GraphEdge("elec_1", "h2_out", "lp_1", "h2_in", FlowType.HYDROGEN))
    # ATR -> LP Tank (Same tank -> Shared)
    adapter.add_edge(GraphEdge("atr_1", "h2_out", "lp_1", "h2_in", FlowType.HYDROGEN))
    
    # Convert
    config_dict = adapter.to_config_dict()
    print("Generated Config Keys:", config_dict.keys())
    
    # Verify
    assert config_dict["production"]["electrolyzer"]["max_power_mw"] == 10.0
    assert config_dict["production"]["atr"]["max_ng_flow_kg_h"] == 200.0
    assert config_dict["storage"]["lp_tanks"]["count"] == 5
    assert config_dict["storage"]["source_isolated"] is False # Shared storage
    assert config_dict["battery"]["capacity_kwh"] == 2000.0
    assert config_dict["water_treatment"]["treatment_block"]["max_flow_m3h"] == 15.0
    
    print("Full plant adapter test passed!")
    return config_dict

def test_plant_builder(config_dict):
    print("\nTesting PlantBuilder.from_dict with full config...")
    try:
        plant = PlantBuilder.from_dict(config_dict)
        print("Plant built successfully!")
        print(f"Registry contains: {plant.registry.get_all_ids()}")
        
        # Verify components exist
        assert plant.registry.has("battery")
        assert plant.registry.has("water_treatment_block")
        assert plant.registry.has("water_pump_a")
        
    except Exception as e:
        print(f"PlantBuilder failed: {e}")
        raise

if __name__ == "__main__":
    try:
        config = test_adapter_full_plant()
        test_plant_builder(config)
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
