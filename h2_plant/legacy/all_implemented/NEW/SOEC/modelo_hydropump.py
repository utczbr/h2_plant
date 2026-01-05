# modelo_hydropump.py (SUGESTÃO DE CORREÇÃO BASEADA EM COOLPROP)
import CoolProp.CoolProp as CP
import numpy as np

def calcular_bomba_modular(P1_kPa: float, T1_C: float, P2_kPa: float, Vazao_m_kgs: float, h_in_kJ_kg: float, Eta_is: float, Eta_m: float, Eta_el: float) -> dict:
    """
    Simula uma bomba hidráulica usando CoolProp para compressão de líquido.
    
    P1_kPa, P2_kPa: Pressões em kPa.
    T1_C: Temperatura de entrada em °C.
    Vazao_m_kgs: Vazão mássica em kg/s.
    h_in_kJ_kg: Entalpia de entrada em kJ/kg.
    """
    T1_K = T1_C + 273.15
    P1_Pa = P1_kPa * 1000.0
    P2_Pa = P2_kPa * 1000.0

    # 🛑 REMOVIDO BLOCO TRY/EXCEPT COM FALLBACK
    
    # 🛑 CORREÇÃO CRÍTICA: Re-calcular h1 usando T1 e P1 no CoolProp
    # Isso garante que a entalpia esteja na base de referência correta, eliminando o erro H molar.
    h1_J_kg = CP.PropsSI('H', 'T', T1_K, 'P', P1_Pa, 'Water')

    # 1. Vazão volumétrica (CoolProp: D - Densidade)
    rho_D1 = CP.PropsSI('D', 'T', T1_K, 'P', P1_Pa, 'Water')
    
    # 2. Trabalho Isentrópico (ideal) no fluido: w_s = V * dP
    v1 = 1.0 / rho_D1 # Volume específico (m3/kg)
    w_s_J_kg = v1 * (P2_Pa - P1_Pa) # J/kg
    
    # 3. Entalpia e Estado de Saída Isentrópico
    h2s_J_kg = h1_J_kg + w_s_J_kg
    T2s_K = CP.PropsSI('T', 'H', h2s_J_kg, 'P', P2_Pa, 'Water')
    
    # 4. Trabalho Real e Entalpia de Saída Real
    w_real_J_kg = w_s_J_kg / Eta_is
    h2_J_kg = h1_J_kg + w_real_J_kg
    
    # 5. Estado de Saída Real
    T2_K = CP.PropsSI('T', 'H', h2_J_kg, 'P', P2_Pa, 'Water')
    T2_C = T2_K - 273.15
    
    # 6. Potências
    Pot_Fluido_kW = (Vazao_m_kgs * w_real_J_kg) / 1000.0 # Ẇ_fluido
    Pot_Eixo_kW = Pot_Fluido_kW / Eta_m # Ẇ_eixo
    Pot_Eletrica_kW = Pot_Eixo_kW / Eta_el # Ẇ_eletrico
    
    return {
        'W_is_J_kg': w_s_J_kg,
        'W_real_kJ_kg': w_real_J_kg / 1000.0,
        'Pot_Fluido_kW': Pot_Fluido_kW,
        'Pot_Eixo_kW': Pot_Eixo_kW,
        'Pot_Eletrica_kW': Pot_Eletrica_kW,
        'T_out_C': T2_C,
        'P_out_kPa': P2_kPa,
        'h_out_kJ_kg': h2_J_kg / 1000.0,
        'erro': None
    }