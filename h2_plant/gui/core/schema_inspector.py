"""
Dynamic schema inspection for widget generation.

Reads plant_schema_v1.json and exposes validation rules to GUI.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

class SchemaInspector:
    """
    Provides validation constraints from JSON schema.
    
    The GUI uses this to generate widgets with proper constraints
    (min/max, enums, type conversions, etc.).
    """
    
    def __init__(self, schema_path: Optional[Path] = None):
        if schema_path is None:
            # Default to h2_plant's bundled schema
            # Assuming this file is in h2_plant/gui/core/
            # Schema is in h2_plant/config/schemas/plant_schema_v1.json
            schema_path = (
                Path(__file__).parent.parent.parent / 
                "config" / "schemas" / "plant_schema_v1.json"
            )
        
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
    
    def get_node_schema(self, node_type: str) -> Dict[str, Any]:
        """
        Get schema for a specific node type.
        
        Example:
            schema = inspector.get_node_schema("ElectrolyzerNode")
            # Returns schema for production.electrolyzer properties
        """
        mapping = {
            "ElectrolyzerNode": ["production", "properties", "electrolyzer"],
            "ATRSourceNode": ["production", "properties", "atr"],
            "LPTankNode": ["storage", "properties", "lp_tanks"],
            "HPTankNode": ["storage", "properties", "hp_tanks"],
            "FillingCompressorNode": ["compression", "properties", "filling_compressor"],
            "OutgoingCompressorNode": ["compression", "properties", "outgoing_compressor"],
            "DemandSchedulerNode": ["demand"],
            "EnergyPriceNode": ["energy_price"],
        }
        
        path = mapping.get(node_type)
        if not path:
            return {}
        
        current = self.schema
        # Schema structure might be definitions based or direct
        # For now assuming standard JSON schema structure where properties are nested
        # Adjust traversal based on actual schema structure if needed
        
        # If the schema uses "definitions", we might need to resolve refs
        # But let's assume the path provided matches the schema structure
        
        try:
            for part in path:
                if "properties" in current:
                    current = current["properties"].get(part, {})
                else:
                    current = current.get(part, {})
            return current
        except (KeyError, AttributeError):
            return {}
    
    def get_property_validator(self, node_type: str, property_name: str) -> Dict[str, Any]:
        """
        Get validation rules for a specific property.
        
        Returns a dict with:
            - type: "number", "integer", "string", "boolean"
            - minimum, maximum
            - enum (if applicable)
            - pattern (regex for strings)
            - description
        """
        node_schema = self.get_node_schema(node_type)
        properties = node_schema.get("properties", {})
        return properties.get(property_name, {})
    
    def list_required_properties(self, node_type: str) -> List[str]:
        """Get list of required properties for a node type."""
        node_schema = self.get_node_schema(node_type)
        return node_schema.get("required", [])
    
    def get_enum_values(self, node_type: str, property_name: str) -> Tuple[str, ...]:
        """Get possible values for an enum property."""
        validator = self.get_property_validator(node_type, property_name)
        return tuple(validator.get("enum", []))
    
    def validate_property(self, 
                         node_type: str, 
                         property_name: str, 
                         value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a property value against schema.
        
        Returns (is_valid, error_message)
        """
        validator = self.get_property_validator(node_type, property_name)
        
        if not validator:
            # If no validator found, assume valid or error? 
            # For now, if it's not in schema, we can't validate it.
            return True, None 
        
        prop_type = validator.get("type")
        
        # Type validation
        if prop_type == "number":
            try:
                float(value)
            except (ValueError, TypeError):
                return False, f"Expected number, got {type(value).__name__}"
            
            val = float(value)
            minimum = validator.get("minimum")
            if minimum is not None and val < minimum:
                return False, f"Value {val} < minimum {minimum}"
            
            exclusive_min = validator.get("exclusiveMinimum")
            if exclusive_min is not None:
                 # JSON schema exclusiveMinimum can be boolean or number
                 if isinstance(exclusive_min, bool) and exclusive_min and val <= minimum:
                     return False, f"Value {val} <= {minimum}"
                 elif isinstance(exclusive_min, (int, float)) and val <= exclusive_min:
                     return False, f"Value {val} <= {exclusive_min}"
            
            maximum = validator.get("maximum")
            if maximum is not None and val > maximum:
                return False, f"Value {val} > maximum {maximum}"
        
        elif prop_type == "integer":
            try:
                int(value)
            except (ValueError, TypeError):
                return False, f"Expected integer, got {type(value).__name__}"
            
            val = int(value)
            minimum = validator.get("minimum")
            if minimum is not None and val < minimum:
                return False, f"Value {val} < minimum {minimum}"
        
        elif prop_type == "string":
            if not isinstance(value, str):
                return False, f"Expected string, got {type(value).__name__}"
        
        # Enum validation
        enum_values = validator.get("enum")
        if enum_values and value not in enum_values:
            return False, f"Value '{value}' not in {enum_values}"
        
        return True, None
