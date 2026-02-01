import pytest
import numpy as np
from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.models.soec_operation import simular_passo_soec, atualizar_mapa_virtual
from h2_plant.config.constants_physics import SOECConstants

CONST = SOECConstants()

def test_soec_parity():
    # Configuration
    config = {
        "num_modules": 6,
        "max_power_nominal_mw": 2.4,
        "optimal_limit": 0.80,
        "rotation_enabled": True,
        "degradation_year": 0.0,
        "real_off_modules": []
    }
    
    physics_config = {
        "soec": {
            "power_first_step_mw": 0.12,
            "ramp_step_mw": 0.24
        }
    }
    
    # Initialize New Component
    soec_new = SOECOperator(config, physics_config)
    
    # Initialize Reference State (Manual Setup to match)
    # Reference uses global/module level vars, we need to mimic the loop
    # We will use the 'simular_passo_soec' function which is the core of the reference
    
    # Initial State for Reference
    n_mod = 6
    potencias = np.full(n_mod, 0.0) # Standby is 0.0 in reference init usually, but let's check
    # Reference init: current_real_powers = np.full(NUM_MODULES, REAL_STAND_BY_POWER, dtype=float) -> 0.0
    estados = np.full(n_mod, 1, dtype=int)
    limites = np.full(n_mod, 2.4 * 0.80)
    mapa_virtual = np.arange(n_mod)
    
    # Test Sequence
    setpoints = [1.0, 5.0, 10.0, 12.0, 8.0, 2.0, 0.5]
    
    print("\nStarting Parity Test...")
    
    for i, sp in enumerate(setpoints):
        # --- Run New Component ---
        p_new, h2_new, steam_new = soec_new.step(sp)
        
        # --- Run Reference Logic ---
        # Rotation
        if i > 0 and i % 60 == 0: # Rotation period 60 min
             mapa_virtual = atualizar_mapa_virtual(mapa_virtual)
             
        # Step
        potencias, estados, limites, mapa_virtual, p_ref_total = simular_passo_soec(
            potencia_referencia=sp,
            potencias_atuais_reais=potencias,
            estados_atuais_reais=estados,
            limites_reais=limites,
            mapa_virtual=mapa_virtual,
            rotacao_ativada=True,
            modulos_desligados_reais=[],
            potencia_limite_eficiente=True,
            year=0.0
        )
        
        # Compare Total Power
        diff = abs(p_new - p_ref_total)
        print(f"Step {i}: SP={sp} MW | New={p_new:.4f} | Ref={p_ref_total:.4f} | Diff={diff:.6f}")
        
        assert diff < 1e-3, f"Power mismatch at step {i}: {diff}"
        
        # Compare Individual Powers (sorted to ignore rotation mapping diffs if any, though they should match)
        # Actually, if logic is identical, real arrays should match exactly index-by-index
        # But let's check sorted first to be safe against mapping implementation details
        # Wait, mapping should be identical too.
        
        np.testing.assert_allclose(soec_new.real_powers, potencias, atol=1e-3, err_msg=f"Module powers mismatch at step {i}")
        
    print("Parity Test Passed!")

if __name__ == "__main__":
    test_soec_parity()
