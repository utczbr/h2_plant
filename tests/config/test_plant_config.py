import pytest
from h2_plant.config.plant_config import PlantConfig, ElectrolyzerConfig, ProductionConfig, ATRConfig, SourceIsolatedStorageConfig, TankArrayConfig


def test_electrolyzer_config_validation():
    """Test electrolyzer configuration validation."""
    
    # Valid configuration
    config = ElectrolyzerConfig(max_power_mw=2.5, base_efficiency=0.65)
    config.validate()  # Should not raise
    
    # Invalid efficiency
    invalid_config = ElectrolyzerConfig(max_power_mw=2.5, base_efficiency=1.5)
    with pytest.raises(ValueError, match="Efficiency"):
        invalid_config.validate()


def test_plant_config_cross_validation():
    """Test cross-validation between subsystems."""
    
    # Source-isolated storage without ATR tanks should fail
    config = PlantConfig()
    config.storage.source_isolated = True
    config.production.atr = ATRConfig(max_ng_flow_kg_h=100.0)
    config.storage.isolated_config = SourceIsolatedStorageConfig(
        electrolyzer_tanks=TankArrayConfig(count=1)
        # Missing atr_tanks!
    )
    
    with pytest.raises(ValueError, match="ATR configured but no ATR tanks"):
        config.validate()
