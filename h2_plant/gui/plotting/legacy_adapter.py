import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from h2_plant.core.component_registry import ComponentRegistry

class LegacyDataAdapter:
    """
    Adapter to convert SimulationEngine results into legacy DataFrame formats.
    
    This class bridges the gap between the new object-oriented component architecture
    and the legacy procedural plotting scripts. It reconstructs the 'df_h2' and 'df_o2'
    pandas DataFrames with exact column names expected by the plots.
    """
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        
    def generate_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate df_h2 and df_o2 compatible with legacy plots.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_h2, df_o2)
        """
        # Define component sequence for each line
        # These IDs must match what is defined in the topology/registry
        h2_ids = ['H2_Source', 'KOD_1', 'DryCooler_1', 'Chiller_1', 'KOD_2', 'Coalescer_1', 'Chiller_2', 'Deoxo_1', 'PSA_1']
        o2_ids = ['source_o2', 'kod1_o2', 'dry_cooler_o2', 'chiller_o2', 'kod2_o2', 'coalescer_o2', 'valve_o2']
        
        # In case the registry uses different IDs, try to discover them or use defaults
        # For now, we assume standard IDs or we might need a mapping config.
        # Let's check what's in the registry if possible, but here we assume topology.
        
        df_h2 = self._build_line_df(h2_ids, 'H2')
        df_o2 = self._build_line_df(o2_ids, 'O2')
        
        return df_h2, df_o2

    def _build_line_df(self, comp_ids: List[str], fluid_type: str) -> pd.DataFrame:
        rows = []
        
        # Expected legacy columns
        columns = [
            'Componente', 'T_C', 'P_bar', 'mdotgaskgs', 'Agua_Condensada_kg_s',
            'Q_dot_fluxo_W', 'W_dot_comp_W', 'y_H2', 'y_O2', 'y_H2O', 'w_H2O', 'H_mix_J_kg'
        ]
        
        for comp_id in comp_ids:
            if not self.registry.has(comp_id):
                continue
                
            comp = self.registry.get(comp_id)
            if not comp:
                continue
                
            state = comp.get_state()
            
            # Extract common properties
            # Note: state keys are lowercase/snake_case in new system
            # Legacy expects specific headers
            
            # Map Component Name
            name = comp_id
            comp_id_lower = comp_id.lower()
            
            if 'kod' in comp_id_lower: 
                # Extract number from ID like KOD_1 -> 1
                parts = comp_id.replace('-', '_').split('_')
                suffix = next((p for p in reversed(parts) if p.isdigit()), "")
                name = f"KOD {suffix}" if suffix else "KOD"
                
            if 'chiller' in comp_id_lower: 
                parts = comp_id.replace('-', '_').split('_')
                suffix = next((p for p in reversed(parts) if p.isdigit()), "1")
                name = f"Chiller {suffix}"
                
            if 'drycooler' in comp_id_lower or 'dry_cooler' in comp_id_lower: 
                name = "Dry Cooler 1"
                
            if 'coalescer' in comp_id_lower: name = "Coalescedor 1"
            if 'deoxo' in comp_id_lower: name = "Deoxo"
            if 'psa' in comp_id_lower: name = "PSA"
            if 'heater' in comp_id_lower: name = "Aquecedor"
            if 'source' in comp_id_lower: name = "Entrada"
            
            # 1. Thermodynamics
            t_c = state.get('temperature_k', 273.15) - 273.15
            p_bar = state.get('pressure_pa', 1e5) / 1e5
            
            # 2. Flow
            # New system output is usually total mass flow.
            # Legacy distinguishes 'mdotgaskgs' (gas only?) vs 'Agua'
            # We use total flow for now or separate if component exposes fractions
            m_total_kg_h = state.get('mass_flow_kg_h', 0.0)
            m_total_kg_s = m_total_kg_h / 3600.0
            
            # Composition & Properties
            y_h2 = 0.0
            y_o2 = 0.0
            y_h2o = 0.0
            w_h2o = 0.0
            h_mix = 0.0
            
            # Molecular Weights (kg/mol)
            MW_H2 = 2.016e-3
            MW_O2 = 32.00e-3
            MW_H2O = 18.015e-3
            MW_N2 = 28.013e-3
            
            stream = self._try_get_output(comp, ['outlet', 'gas_out', 'fluid_out', 'h2_out', 'gas_outlet', 'purified_gas_out'])
            if not stream and hasattr(comp, 'output_stream'):
                stream = comp.output_stream
                
            if stream:
                comp_dict = stream.composition
                y_h2 = comp_dict.get('H2', 0.0)
                y_o2 = comp_dict.get('O2', 0.0)
                y_h2o = comp_dict.get('H2O', 0.0)
                y_n2 = comp_dict.get('N2', 0.0)
                
                # Calculate MW_mix
                mw_mix = (y_h2 * MW_H2 + y_o2 * MW_O2 + y_h2o * MW_H2O + y_n2 * MW_N2)
                if mw_mix > 0:
                    # Mass fractions: w_i = (y_i * MW_i) / MW_mix
                    w_h2o = (y_h2o * MW_H2O) / mw_mix
                
                # Enthalpy Calculation (approximate as ideal mixture)
                # H_mix = sum(w_i * H_i)
                # We need H_i(T, P)
                # Ideally use LUTManager, but for simplicity here use CoolProp fallback if needed
                # or just use CP directly if we don't want to inject LUTManager.
                # Since we are in an adapter, let's try to get LUT from registry if available.
                
                lut = None
                if self.registry.has('lut_manager'):
                    lut = self.registry.get('lut_manager')
                    
                if lut:
                    # Use LUT
                    h_h2 = lut.lookup('H2', 'H', stream.pressure_pa, stream.temperature_k)
                    h_o2 = lut.lookup('O2', 'H', stream.pressure_pa, stream.temperature_k)
                    h_h2o = lut.lookup('H2O', 'H', stream.pressure_pa, stream.temperature_k) # Vapor?
                    h_n2 = lut.lookup('N2', 'H', stream.pressure_pa, stream.temperature_k)
                else:
                    # Fallback or zero
                    h_h2 = h_o2 = h_h2o = h_n2 = 0.0
                    
                # Mass fractions for mixing enthalpy
                if mw_mix > 0:
                    w_h2 = (y_h2 * MW_H2) / mw_mix
                    w_o2 = (y_o2 * MW_O2) / mw_mix
                    w_n2 = (y_n2 * MW_N2) / mw_mix
                    
                    h_mix = (w_h2 * h_h2 + w_o2 * h_o2 + w_h2o * h_h2o + w_n2 * h_n2)

                # UPDATE T/P FROM STREAM (Prioritize actual output over state default)
                t_c = stream.temperature_k - 273.15
                p_bar = stream.pressure_pa / 1e5

            # 3. Energy & Water
            q_dot_w = state.get('heat_transfer_w', 0.0)
            if 'cooling_load_kw' in state:
                q_dot_w = state['cooling_load_kw'] * 1000.0
                
            w_dot_w = state.get('power_consumption_kw', 0.0) * 1000.0
            
            water_rem_kg_h = state.get('water_removed_kg_h', 0.0)
            
            # Try to get water removal from actual drain streams if state is missing it
            if water_rem_kg_h <= 0:
                drain_stream = self._try_get_output(comp, ['liquid_drain', 'drain'])
                if drain_stream:
                    water_rem_kg_h = drain_stream.mass_flow_kg_h
            
            water_rem_kg_s = water_rem_kg_h / 3600.0
            
            row = {
                'Componente': name,
                'T_C': t_c,
                'P_bar': p_bar,
                'mdotgaskgs': m_total_kg_s,
                'Agua_Condensada_kg_s': water_rem_kg_s,
                'Q_dot_fluxo_W': q_dot_w,
                'W_dot_comp_W': w_dot_w,
                'y_H2': y_h2,
                'y_O2': y_o2,
                'y_H2O': y_h2o,
                'w_H2O': w_h2o,
                'H_mix_J_kg': h_mix
            }
            
            rows.append(row)
            
        if not rows:
            return pd.DataFrame(columns=columns)
            
        return pd.DataFrame(rows)

    def _try_get_output(self, comp: Any, port_names: List[str]) -> Any:
        """Try to get output from a list of ports, returning the first success or None."""
        for port in port_names:
            try:
                # Some components raise errors for unknown ports, others return None
                # We handle both cases here
                out = comp.get_output(port)
                if out is not None:
                    # DEBUG LOG
                    # print(f"DEBUG: Found output for {getattr(comp, 'component_id', 'Unk')} on port {port}")
                    return out
            except (ValueError, KeyError, NotImplementedError):
                continue
            except Exception as e:
                # Catch-all for other unexpected component errors
                print(f"DEBUG: Error getting output for {getattr(comp, 'component_id', 'Unk')} port {port}: {e}")
                continue
        # print(f"DEBUG: Failed to get output for {getattr(comp, 'component_id', 'Unk')} from {port_names}")
        return None
