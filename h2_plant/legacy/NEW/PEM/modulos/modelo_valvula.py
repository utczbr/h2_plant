# Valvula.py (Conteúdo Original - Está Correto em sua Lógica Termodinâmica)
import CoolProp.CoolProp as CP

def modelo_valvula_isoentalpica(fluido, T_in_K, P_in_Pa, P_out_Pa):
    """
    Modela o processo de estrangulamento (isoentálpico) de uma válvula simples
    usando o CoolProp.
    ... (omiti docstrings e exemplos para brevidade) ...
    """
    
    # 1. Verificação de Condição (Estrangulamento requer P_out < P_in)
    if P_out_Pa >= P_in_Pa:
        print("Erro: A pressão de saída (P_out) deve ser menor que a pressão de entrada (P_in) para um processo de estrangulamento (isoentálpico/adiabático).")
        return None

    try:
        # --- ENTRADA ---
        # Calcular a entalpia de entrada (H_in)
        H_in_J_kg = CP.PropsSI('H', 'T', T_in_K, 'P', P_in_Pa, fluido)

        # --- SAÍDA (Processo Isoentálpico: H_out = H_in) ---
        H_out_J_kg = H_in_J_kg

        # Calcular a Temperatura de Saída (T_out) usando a nova Pressão (P_out) 
        T_out_K = CP.PropsSI('T', 'P', P_out_Pa, 'H', H_out_J_kg, fluido)

        # ... (cálculo de S_out, D_out omitido para brevidade no retorno) ...

        resultados = {
            "Fluido": fluido,
            "ENTRADA": {
                "T_K": T_in_K,
                "P_Pa": P_in_Pa,
                "H_J_kg": H_in_J_kg,
            },
            "SAIDA": {
                "T_K": T_out_K,
                "P_Pa": P_out_Pa,
                "H_J_kg": H_out_J_kg,
            }
        }
        
        return resultados

    except ValueError as e:
        print(f"Erro de cálculo no CoolProp: {e}")
        print("Verifique se as condições de entrada (T, P) estão dentro dos limites do CoolProp para o fluido selecionado.")
        return None