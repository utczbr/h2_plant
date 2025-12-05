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
    
    def __post_init__(self):
        """Validate composition."""
        total_fraction = sum(self.composition.values())
        if abs(total_fraction - 1.0) > 1e-3 and total_fraction > 0:
            # Normalize if not summing to 1 (unless empty)
            for k in self.composition:
                self.composition[k] /= total_fraction

    @property
    def specific_enthalpy_j_kg(self) -> float:
        """
        Calculate specific enthalpy (J/kg) relative to reference state (298.15K).
        Assumes ideal gas mixture behavior for gases.
        h = sum(xi * hi) where hi = integral(Cp_i * dT)
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
                # Wait, Shomate equation usually gives J/mol/K for Cp, so integral is J/mol.
                # Need to convert to J/kg.
                # Let's verify units. Standard NIST Shomate gives J/mol*K.
                # So result is J/mol. Divide by MW (g/mol) -> J/g -> *1000 -> J/kg.
                # Actually GasConstants usually stores MW in g/mol.
                
                # Correction: GasConstants.SPECIES_DATA usually has MW in g/mol.
                # Result of integral is J/mol.
                # J/mol / (g/mol) = J/g.
                # J/g * 1000 = J/kg.
                
                h_species_j_kg = (integral_cp(self.temperature_k) - integral_cp(t_ref)) * 1000.0 / data['molecular_weight']
                h_total += fraction * h_species_j_kg
                
        return h_total

    @property
    def density_kg_m3(self) -> float:
        """
        Calculate density (kg/m3).
        Assumes ideal gas law for gases: rho = P / (R_specific * T)
        """
        if self.phase == 'liquid':
            # Simplified liquid density (mostly water)
            if 'H2O' in self.composition and self.composition['H2O'] > 0.9:
                return 1000.0 # approx for water
            return 1000.0 # fallback
            
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
            phase=self.phase # Assume phase matches for now
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
