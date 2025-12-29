"""
Stream class for thermodynamic resource tracking.

Represents a material flow with intrinsic properties:
- Mass flow (kg/h)
- Temperature (K)
- Pressure (Pa)
- Composition (mass fractions)
- Thermodynamic state (Enthalpy, Entropy, Density)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
from h2_plant.core.constants import GasConstants, StandardConditions, ConversionFactors

@dataclass
class Stream:
    """
    Represents a material flow with thermodynamic properties.
    """
    mass_flow_kg_h: float
    temperature_k: float = StandardConditions.TEMPERATURE_K
    pressure_pa: float = StandardConditions.PRESSURE_PA
    composition: Dict[str, float] = field(default_factory=lambda: {'H2': 1.0})
    phase: str = 'gas'  # 'gas', 'liquid', 'mixed'
    extra: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate composition and cache expensive properties."""
        total_fraction = sum(self.composition.values())
        if abs(total_fraction - 1.0) > 1e-3 and total_fraction > 0:
            # Normalize if not summing to 1 (unless empty)
            for k in self.composition:
                self.composition[k] /= total_fraction
        
        # PERFORMANCE: Cache enthalpy calculation (called ~47k times per simulation)
        self._cached_enthalpy: float = self._compute_specific_enthalpy()

    def _compute_specific_enthalpy(self) -> float:
        """
        Internal method to compute specific enthalpy (J/kg).
        Called once at construction and cached.
        """
        h_total = 0.0
        t_ref = StandardConditions.TEMPERATURE_K
        
        for species, fraction in self.composition.items():
            if species in GasConstants.SPECIES_DATA:
                data = GasConstants.SPECIES_DATA[species]
                # Cp(T) = A + B*T + C*T^2 + D*T^3 + E/T^2
                # H(T) - H(Tref) = integral from Tref to T of Cp(T) dT
                # Integral = A*T + B*T^2/2 + C*T^3/3 + D*T^4/4 - E/T
                
                coeffs = data.get('cp_coeffs', [0, 0, 0, 0, 0])
                A, B, C, D, E = coeffs
                
                def integral_cp(t):
                    return (A * t + 
                            B * t**2 / 2 + 
                            C * t**3 / 3 + 
                            D * t**4 / 4 - 
                            E / t) 
                
                h_species_j_kg = (integral_cp(self.temperature_k) - integral_cp(t_ref)) * 1000.0 / data['molecular_weight']
                h_total += fraction * h_species_j_kg
                
        return h_total

    @property
    def entrained_liq_kg_s(self) -> float:
        """Entrained H2O liquid carryover (kg/s); 0 if pure vapor/gas."""
        return self.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0)

    @property
    def mole_fractions(self) -> Dict[str, float]:
        """
        Compute mole fractions from mass fractions.
        y_i = (x_i / MW_i) / sum(x_j / MW_j)
        """
        moles_rel = {}
        total_moles = 0.0
        
        for s, mass_frac in self.composition.items():
            if mass_frac <= 0:
                continue
            mw = GasConstants.SPECIES_DATA.get(s, {}).get('molecular_weight', 28.0) # Default to N2 if unknown
            moles = mass_frac / mw
            moles_rel[s] = moles
            total_moles += moles
            
        if total_moles <= 0:
            return {}
            
        return {s: m / total_moles for s, m in moles_rel.items()}

    def get_mole_frac(self, species: str) -> float:
        """Helper to get mole fraction of a specific species."""
        return self.mole_fractions.get(species, 0.0)

    def get_total_mole_frac(self, species: str) -> float:
        """
        Get mole fraction including extra liquid water in total moles.
        
        This matches the calculation used in the Stream Summary Table,
        which includes entrained liquid water from the 'extra' dict.
        
        Args:
            species: Species name (e.g., 'O2', 'H2', 'H2O')
            
        Returns:
            Mole fraction of the species including all water in denominator.
        """
        # Molecular weights (g/mol)
        MW_H2 = 2.016
        MW_H2O = 18.015
        MW_O2 = 32.0
        
        m_dot_main = self.mass_flow_kg_h
        if m_dot_main <= 0:
            return 0.0
        
        # Mass of each species (kg/h)
        m_h2 = self.composition.get('H2', 0.0) * m_dot_main
        m_o2 = self.composition.get('O2', 0.0) * m_dot_main
        
        # Water: vapor + liquid (from composition) + extra liquid
        m_h2o_vapor = self.composition.get('H2O', 0.0) * m_dot_main
        m_h2o_liq_comp = self.composition.get('H2O_liq', 0.0) * m_dot_main
        
        # Extra liquid water (convert kg/s to kg/h)
        m_dot_extra_liq = self.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0) * 3600.0
        
        # Total water: ALWAYS include all sources (vapor + composition liquid + extra liquid)
        m_h2o_total = m_h2o_vapor + m_h2o_liq_comp + m_dot_extra_liq
        
        # Convert to moles
        n_h2 = m_h2 / MW_H2
        n_h2o = m_h2o_total / MW_H2O
        n_o2 = m_o2 / MW_O2
        n_total = n_h2 + n_h2o + n_o2
        
        if n_total <= 0:
            return 0.0
        
        # Return mole fraction of requested species
        if species == 'H2':
            return n_h2 / n_total
        elif species == 'O2':
            return n_o2 / n_total
        elif species in ('H2O', 'H2O_liq'):
            return n_h2o / n_total
        else:
            # Fallback to the basic mole_fractions for other species
            return self.mole_fractions.get(species, 0.0)

    @property
    def specific_enthalpy_j_kg(self) -> float:
        """
        Return cached specific enthalpy (J/kg) relative to reference state (298.15K).
        Computed once at Stream construction for performance.
        """
        return self._cached_enthalpy

    @property
    def specific_entropy_j_kgK(self) -> float:
        """
        Calculate specific entropy (J/kg·K) relative to reference state (298.15K, 1 atm).
        Includes mixing entropy term.
        
        s_mix_molar = sum(yi * s_i_molar(T, P)) - R * sum(yi * ln(yi))
        Where s_i_molar(T, P) = s_i_standard(T) - R * ln(P/Pref)
        
        Final s_specific = s_mix_molar / M_mix
        """
        if self.mass_flow_kg_h <= 0:
            return 0.0

        t_calc = max(self.temperature_k, 1.0)
        t_ref = StandardConditions.TEMPERATURE_K
        p_ref = StandardConditions.PRESSURE_PA
        r_gas = GasConstants.R_UNIVERSAL_J_PER_MOL_K
        
        # 1. Calculate Mole Fractions and M_mix
        moles_relative = {}
        total_moles_rel = 0.0
        
        # Identify valid species
        valid_species = []
        for species, mass_frac in self.composition.items():
            if mass_frac > 0 and species in GasConstants.SPECIES_DATA:
                mw = GasConstants.SPECIES_DATA[species]['molecular_weight'] # g/mol
                moles = mass_frac / mw
                moles_relative[species] = moles
                total_moles_rel += moles
                valid_species.append(species)
        
        if total_moles_rel <= 0:
            return 0.0
            
        m_mix_g_mol = 1.0 / total_moles_rel
        
        # 2. Calculate Molar Entropy
        s_mix_molar = 0.0
        
        # Pressure correction term (ideal gas: -R ln(P/Pref)) applies to all
        
        pressure_term = -r_gas * np.log(self.pressure_pa / p_ref)
        mix_term = 0.0
        standard_term = 0.0
        
        for species in valid_species:
            yi = moles_relative[species] / total_moles_rel
            data = GasConstants.SPECIES_DATA[species]
            coeffs = data.get('cp_coeffs', [0, 0, 0, 0, 0])
            A, B, C, D, E = coeffs
            
            # Integral Cp/T dT from Tref to T
            # Cp/T = A/T + B + C*T + D*T^2 + E/T^3
            # Int = A ln(T) + B*T + C*T^2/2 + D*T^3/3 - E/(2*T^2)
            def integral_cp_t(t):
                return (A * np.log(t) + 
                        B * t + 
                        C * t**2 / 2 + 
                        D * t**3 / 3 - 
                        E / (2 * t**2))
            
            ds_standard = integral_cp_t(t_calc) - integral_cp_t(t_ref)
            
            standard_term += yi * ds_standard
            mix_term -= yi * np.log(yi) # - sum yi ln yi

        # Total Molar Entropy (J/mol·K)
        # Note: We are calculating DELTA S from reference T_ref, P_ref.
        # Absolute entropy would need S_ref values. 
        # For trend plotting, Delta S is fine.
        s_molar_total = standard_term + pressure_term + (r_gas * mix_term)
        
        # Convert to specific entropy (J/kg·K)
        # J/mol·K / (g/mol / 1000) = J/kg·K
        s_specific = s_molar_total * 1000.0 / m_mix_g_mol
        
        return s_specific

    @property
    def density_kg_m3(self) -> float:
        """
        Calculate density (kg/m3).
        Assumes ideal gas law for gases: rho = P / (R_specific * T)
        """
        if self.phase == 'liquid':
            # Attempt LUTManager lookup for accurate water density
            try:
                try:
                    from h2_plant.core.component_registry import ComponentRegistry
                    registry = ComponentRegistry.get_instance() if hasattr(ComponentRegistry, 'get_instance') else None
                    if registry:
                        lut_mgr = registry.get('LUT_MANAGER')
                        if lut_mgr and 'H2O' in self.composition:
                            # Lookup water density at T, P
                            return lut_mgr.lookup('Water', 'D', self.pressure_pa, self.temperature_k)
                except ImportError:
                     pass # Registry not available (standalone mode)
            except Exception:
                pass  # Fallback to hardcoded value
            
            # Simplified liquid density (fallback)
            return 1000.0  # approx for water
            
        # Gas phase
        # Calculate mixture specific gas constant
        # R_mix = sum(xi * Ri)
        r_mix = 0.0
        for species, fraction in self.composition.items():
            if species in GasConstants.SPECIES_DATA:
                mw = GasConstants.SPECIES_DATA[species]['molecular_weight'] # g/mol
                # R_specific = R_universal / MW
                # R_univ = 8.314 J/(mol K)
                # MW in kg/mol = MW_g_mol / 1000
                # R_spec = 8.314 / (MW/1000) = 8314 / MW
                r_spec = 8314.0 / mw
                r_mix += fraction * r_spec
        
        if r_mix > 0:
            return self.pressure_pa / (r_mix * self.temperature_k)
        return 0.0

    @property
    def volume_flow_m3_h(self) -> float:
        """Calculate volume flow rate (m3/h)."""
        rho = self.density_kg_m3
        if rho > 0:
            return self.mass_flow_kg_h / rho
        return 0.0

    def mix_with(self, other: 'Stream') -> 'Stream':
        """
        Mix this stream with another stream.
        Conserves mass and enthalpy.
        Pressure equilibrates to the lower of the two (simplified) or requires external logic.
        Here we assume mixing happens at the lower pressure of the two streams.
        """
        total_mass = self.mass_flow_kg_h + other.mass_flow_kg_h
        if total_mass <= 0:
            return Stream(0.0)
            
        # Mass fractions
        new_comp = {}
        all_species = set(self.composition.keys()) | set(other.composition.keys())
        for s in all_species:
            m1 = self.mass_flow_kg_h * self.composition.get(s, 0.0)
            m2 = other.mass_flow_kg_h * other.composition.get(s, 0.0)
            new_comp[s] = (m1 + m2) / total_mass
            
        # Extra (metadata) mixing - weighted average
        new_extra = {}
        all_extra = set(self.extra.keys()) | set(other.extra.keys())
        for k in all_extra:
             v1 = self.extra.get(k, 0.0) * self.mass_flow_kg_h
             v2 = other.extra.get(k, 0.0) * other.mass_flow_kg_h
             new_extra[k] = (v1 + v2) / total_mass
            
        # Enthalpy conservation
        # H_mix = H1 + H2
        # m_mix * h_mix = m1 * h1 + m2 * h2
        h1 = self.specific_enthalpy_j_kg
        h2 = other.specific_enthalpy_j_kg
        h_mix = (self.mass_flow_kg_h * h1 + other.mass_flow_kg_h * h2) / total_mass
        
        # Find T_mix such that enthalpy(T_mix) = h_mix
        # Iterative solver (Newton-Raphson)
        t_mix = (self.temperature_k * self.mass_flow_kg_h + other.temperature_k * other.mass_flow_kg_h) / total_mass # Initial guess
        
        # Simple iteration to converge T
        # For ideal gases with constant Cp, T_mix is weighted average.
        # With variable Cp, we iterate.
        for _ in range(5):
            temp_stream = Stream(total_mass, t_mix, StandardConditions.PRESSURE_PA, new_comp)
            h_curr = temp_stream.specific_enthalpy_j_kg
            
            # Estimate Cp at current T
            cp_mix = 0.0
            for s, f in new_comp.items():
                if s in GasConstants.SPECIES_DATA:
                    # Simplified Cp estimation (using first coeff A)
                    # Cp ~ A (J/mol/K) -> J/kg/K
                    mw = GasConstants.SPECIES_DATA[s]['molecular_weight']
                    cp_s = GasConstants.SPECIES_DATA[s]['cp_coeffs'][0] * 1000.0 / mw
                    cp_mix += f * cp_s
            
            if cp_mix > 0:
                delta_t = (h_mix - h_curr) / cp_mix
                t_mix += delta_t
            else:
                break
                
        final_pressure = min(self.pressure_pa, other.pressure_pa)
        
        return Stream(
            mass_flow_kg_h=total_mass,
            temperature_k=t_mix,
            pressure_pa=final_pressure,
            composition=new_comp,
            phase=self.phase, # Assume phase matches for now
            extra=new_extra
        )

    def compress_isentropic(self, outlet_pressure_pa: float, isentropic_efficiency: float) -> Tuple['Stream', float]:
        """
        Compress stream to target pressure.
        Returns (new_stream, work_kwh).
        T_out = T_in * (P_out/P_in)^((k-1)/k)
        """
        if self.pressure_pa <= 0 or outlet_pressure_pa <= self.pressure_pa:
            return self, 0.0
            
        # Calculate mixture gamma (k)
        # k = Cp / Cv
        # Cp_mix = sum(xi * Cpi)
        # Cv_mix = Cp_mix - R_mix
        
        cp_mix_molar = 0.0
        r_mix_molar = 8.314
        
        # Molar average
        total_moles = 0.0
        for s, f in self.composition.items():
            if s in GasConstants.SPECIES_DATA:
                mw = GasConstants.SPECIES_DATA[s]['molecular_weight']
                moles = f / mw
                total_moles += moles
                
                # Cp at current T
                # Using just A term for simplicity in gamma calc, or evaluate full polynomial
                coeffs = GasConstants.SPECIES_DATA[s]['cp_coeffs']
                t = self.temperature_k
                cp_molar = coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3 + coeffs[4]/t**2
                cp_mix_molar += moles * cp_molar
        
        if total_moles > 0:
            cp_mix_molar /= total_moles
            
        cv_mix_molar = cp_mix_molar - r_mix_molar
        gamma = cp_mix_molar / cv_mix_molar if cv_mix_molar > 0 else 1.4
        
        # Isentropic temperature rise
        pressure_ratio = outlet_pressure_pa / self.pressure_pa
        exponent = (gamma - 1) / gamma
        t_isentropic = self.temperature_k * (pressure_ratio ** exponent)
        
        # Actual temperature rise
        delta_t_isentropic = t_isentropic - self.temperature_k
        delta_t_actual = delta_t_isentropic / isentropic_efficiency
        t_out = self.temperature_k + delta_t_actual
        
        # Calculate work
        # W = m * (h_out - h_in)
        # Or W = m * Cp * delta_T
        # Let's use enthalpy difference
        out_stream_ideal = Stream(self.mass_flow_kg_h, t_out, outlet_pressure_pa, self.composition, self.phase)
        h_in = self.specific_enthalpy_j_kg
        h_out = out_stream_ideal.specific_enthalpy_j_kg
        
        work_j = self.mass_flow_kg_h * (h_out - h_in) # J/h
        work_kwh = work_j * ConversionFactors.J_TO_KWH
        
        return out_stream_ideal, work_kwh
