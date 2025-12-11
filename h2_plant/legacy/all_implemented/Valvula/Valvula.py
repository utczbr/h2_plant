import CoolProp.CoolProp as CP

def modelo_valvula_isoentalpica(fluido, T_in_K, P_in_Pa, P_out_Pa):
    """
    Modela o processo de estrangulamento (isoentálpico) de uma válvula simples
    usando o CoolProp.

    Args:
        fluido (str): O nome do fluido (ex: "hydrogen", "H2").
        T_in_K (float): Temperatura de entrada em Kelvin (K).
        P_in_Pa (float): Pressão de entrada em Pascal (Pa).
        P_out_Pa (float): Pressão de saída em Pascal (Pa).

    Returns:
        dict: Um dicionário com as propriedades de entrada e saída.
              Retorna None se a pressão de saída for maior ou igual à de entrada,
              ou se ocorrer um erro de cálculo.
    """
    
    # 1. Verificação de Condição (Estrangulamento requer P_out < P_in)
    if P_out_Pa >= P_in_Pa:
        print("Erro: A pressão de saída (P_out) deve ser menor que a pressão de entrada (P_in) para um processo de estrangulamento (isoentálpico/adiabático).")
        return None

    try:
        # --- ENTRADA ---
        # Calcular a entalpia de entrada (H_in)
        H_in_J_kg = CP.PropsSI('H', 'T', T_in_K, 'P', P_in_Pa, fluido)

        # Outras propriedades de entrada para referência
        S_in_J_kgK = CP.PropsSI('S', 'T', T_in_K, 'P', P_in_Pa, fluido)
        D_in_kg_m3 = CP.PropsSI('D', 'T', T_in_K, 'P', P_in_Pa, fluido)

        # --- SAÍDA (Processo Isoentálpico: H_out = H_in) ---
        H_out_J_kg = H_in_J_kg

        # Calcular a Temperatura de Saída (T_out) usando a nova Pressão (P_out) 
        # e a Entalpia Constante (H_out)
        T_out_K = CP.PropsSI('T', 'P', P_out_Pa, 'H', H_out_J_kg, fluido)

        # Outras propriedades de saída
        S_out_J_kgK = CP.PropsSI('S', 'P', P_out_Pa, 'H', H_out_J_kg, fluido)
        D_out_kg_m3 = CP.PropsSI('D', 'P', P_out_Pa, 'H', H_out_J_kg, fluido)

        # Organizar os resultados
        resultados = {
            "Fluido": fluido,
            "ENTRADA": {
                "T_K": T_in_K,
                "P_Pa": P_in_Pa,
                "H_J_kg": H_in_J_kg,
                "S_J_kgK": S_in_J_kgK,
                "D_kg_m3": D_in_kg_m3
            },
            "SAIDA": {
                "T_K": T_out_K,
                "P_Pa": P_out_Pa, # P_out é a pressão definida
                "H_J_kg": H_out_J_kg, # H_out deve ser igual a H_in (isoentálpico)
                "S_J_kgK": S_out_J_kgK,
                "D_kg_m3": D_out_kg_m3
            }
        }
        
        return resultados

    except ValueError as e:
        print(f"Erro de cálculo no CoolProp: {e}")
        print("Verifique se as condições de entrada (T, P) estão dentro dos limites do CoolProp para o fluido selecionado.")
        return None

# --- EXEMPLO DE USO ---
print("--- Simulação de Válvula de Estrangulamento (Hydrogen) ---")

# Dados de entrada
fluido_trabalho = "hydrogen" # Gás hidrogênio (H2)
T_entrada_K = 300.0          # 300 K (cerca de 26.85 °C)
P_entrada_Pa = 1013250.0     # 10.1325 bar (10 atm)
P_saida_Pa = 200000.0        # 2.0 bar

# Executar o modelo
dados_fluxo = modelo_valvula_isoentalpica(
    fluido=fluido_trabalho, 
    T_in_K=T_entrada_K, 
    P_in_Pa=P_entrada_Pa, 
    P_out_Pa=P_saida_Pa
)

# Exibir resultados
if dados_fluxo:
    print(f"\n✅ Resultados da Simulação para {dados_fluxo['Fluido']}:\n")
    print("ESTADO DE ENTRADA (IN):")
    print(f"  Temperatura (T_in): {dados_fluxo['ENTRADA']['T_K']:.2f} K")
    print(f"  Pressão (P_in): {dados_fluxo['ENTRADA']['P_Pa']:.0f} Pa ({dados_fluxo['ENTRADA']['P_Pa']/100000:.2f} bar)")
    print(f"  Entalpia (H_in): {dados_fluxo['ENTRADA']['H_J_kg']:.2f} J/kg")
    
    print("\nESTADO DE SAÍDA (OUT) - ISOENTÁLPICO:")
    print(f"  Temperatura (T_out): {dados_fluxo['SAIDA']['T_K']:.2f} K")
    print(f"  Pressão (P_out): {dados_fluxo['SAIDA']['P_Pa']:.0f} Pa ({dados_fluxo['SAIDA']['P_Pa']/100000:.2f} bar)")
    print(f"  Entalpia (H_out): {dados_fluxo['SAIDA']['H_J_kg']:.2f} J/kg")
    
    print("\n--- Comparação de Entalpia ---")
    print(f"H_out - H_in = {dados_fluxo['SAIDA']['H_J_kg'] - dados_fluxo['ENTRADA']['H_J_kg']:.4f} J/kg (esperado ≈ 0)")