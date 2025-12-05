import CoolProp.CoolProp as CP
import numpy as np

# --- Dados de Exemplo ---
# Exemplo 1: Entrada Conhecida (Fluxo Direto)
EXEMPLO_ENTRADA = {
    'nome': 'Exemplo (Entrada Conhecida)',
    'estado_conhecido': '1', # 1 = Entrada (P1, T1)
    'P1': 101.325, # kPa (Pressão Atmosférica)
    'T1': 20.0,    # °C
    'P_final': 500.0, # P2 kPa (Pressão de Saída Alvo)
    'Vazao_m': 10.0, # kg/s
    'Eta_is': 0.82,  # Eficiência Isoentrópica
    'Eta_m': 0.96    # Eficiência Mecânica
}

# Exemplo 2: Saída Conhecida (Fluxo Reverso)
EXEMPLO_SAIDA = {
    'nome': 'Exemplo (Saída Conhecida - Reverso)',
    'estado_conhecido': '2', # 2 = Saída (P2, T2)
    'P2': 500.0,   # kPa
    'T2': 20.05,  # °C (Ligeiro aumento de temperatura devido ao trabalho real)
    'P_final': 101.325, # P1 kPa (Pressão de Entrada)
    'Vazao_m': 10.0, # kg/s
    'Eta_is': 0.82,
    'Eta_m': 0.96
}

def calcular_propriedades_bomba():
    """
    Calcula o estado desconhecido (Entrada ou Saída) e a potência da bomba
    usando a eficiência isoentrópica e a biblioteca CoolProp.

    Inclui opções para rodar com exemplos prontos ou com entrada manual.
    """
    print("--- Modelo de Bomba de Água Simples (CoolProp e Eficiência Isoentrópica) ---")
    
    # --- 1. Seleção do Modo de Operação ---
    print("\n--- Seleção de Modo ---")
    print("[1] Exemplo: Entrada Conhecida (P1, T1)")
    print("[2] Exemplo: Saída Conhecida (P2, T2)")
    print("[3] Entrada Manual de Dados")
    
    while True:
        modo = input("Selecione o modo de operação (1, 2 ou 3): ").strip()
        if modo in ('1', '2', '3'):
            break
        print("Opção inválida. Por favor, digite 1, 2 ou 3.")

    # --- 2. Coleta/Atribuição de Dados ---

    dados = {}
    
    if modo == '1':
        dados = EXEMPLO_ENTRADA.copy()
        print(f"\nRodando com o {dados['nome']}.")
    elif modo == '2':
        dados = EXEMPLO_SAIDA.copy()
        print(f"\nRodando com o {dados['nome']}.")
    else: # Modo manual (3)
        print("\n--- Entrada Manual de Dados ---")
        
        # 2.1 Escolha do Estado Conhecido (Manualmente)
        while True:
            estado_conhecido = input("Você conhece as propriedades da (E)ntrada [1] ou da (S)aída [2] da bomba? (Digite 1 ou 2): ").strip()
            if estado_conhecido in ('1', '2', 'e', 'E', 's', 'S'):
                dados['estado_conhecido'] = '1' if estado_conhecido in ('1', 'e', 'E') else '2'
                break
            print("Opção inválida. Por favor, digite 1 (Entrada) ou 2 (Saída).")

        # 2.2 Estado de Referência (Manualmente)
        try:
            if dados['estado_conhecido'] == '1':
                print("\n--- Estado de Entrada (1) Conhecido ---")
                dados['P1'] = float(input("Pressão de Entrada P1 [kPa]: "))
                dados['T1'] = float(input("Temperatura de Entrada T1 [°C]: "))
                dados['P_final'] = float(input("Pressão de Saída P2 [kPa] (Pressão alvo da bomba): "))
                
            else:
                print("\n--- Estado de Saída (2) Conhecido ---")
                dados['P2'] = float(input("Pressão de Saída P2 [kPa]: "))
                dados['T2'] = float(input("Temperatura de Saída T2 [°C]: "))
                dados['P_final'] = float(input("Pressão de Entrada P1 [kPa] (Pressão inicial da bomba): "))
                
            dados['Vazao_m'] = float(input("Vazão Mássica m_dot [kg/s]: "))
            dados['Eta_is'] = float(input("Eficiência Isoentrópica eta_is [0.0 a 1.0, ex: 0.8]: "))
            dados['Eta_m'] = float(input("Eficiência Mecânica eta_m [0.0 a 1.0, ex: 0.95]: "))


            if not (0.0 < dados['Eta_is'] <= 1.0) or not (0.0 < dados['Eta_m'] <= 1.0):
                 raise ValueError("As Eficiências devem estar entre 0.0 e 1.0.")


        except ValueError as e:
            print(f"\nERRO na entrada de dados: {e}")
            return

    # Extrai os dados do dicionário 'dados' para variáveis locais para facilitar o código restante
    estado_conhecido = dados['estado_conhecido']
    Vazao_m = dados['Vazao_m']
    Eta_is = dados['Eta_is']
    Eta_m = dados['Eta_m']
    P_final = dados['P_final']
    
    if estado_conhecido == '1':
        P1, T1 = dados['P1'], dados['T1']
    else:
        P2, T2 = dados['P2'], dados['T2']


    # A substância utilizada é água (Water)
    fluido = 'Water' 
    
    # Constantes para conversão de unidades
    T_kelvin_offset = 273.15
    P_pascal_to_kPa = 1000.0
    
    # --- 3. Cálculos para o Estado Conhecido (1 ou 2) ---
    
    if estado_conhecido == '1':
        # Conversão para unidades do CoolProp: K e Pa
        T1_K = T1 + T_kelvin_offset
        P1_Pa = P1 * P_pascal_to_kPa
        P2_Pa = P_final * P_pascal_to_kPa
        
        # 3.1 Propriedades da Entrada (1)
        try:
            h1 = CP.PropsSI('H', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0  # h1 em kJ/kg
            s1 = CP.PropsSI('S', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0  # s1 em kJ/(kg*K)
        except Exception as e:
            print(f"\nERRO ao calcular propriedades do estado 1 (Entrada): {e}")
            return
            
        # 3.2 Propriedades do Estado Isoentrópico de Saída (2s) - s2s = s1
        try:
            # h2s é a entalpia na pressão de saída P2 e entropia s1
            h2s = CP.PropsSI('H', 'P', P2_Pa, 'S', s1 * 1000.0, fluido) / 1000.0 # h2s em kJ/kg
        except Exception as e:
            print(f"\nERRO ao calcular propriedades do estado 2s (Isoentrópico): {e}")
            return

        # 3.3 Cálculo da Entalpia de Saída Real (h2)
        # Eficiência: Eta_is = (h2s - h1) / (h2 - h1)  =>  h2 - h1 = (h2s - h1) / Eta_is
        Trabalho_is = h2s - h1
        Trabalho_real = Trabalho_is / Eta_is
        h2 = h1 + Trabalho_real  # h2 em kJ/kg
        
        # 3.4 Propriedades da Saída Real (2) - Cálculo da Temperatura T2
        try:
            # T2 é a temperatura na pressão P2 e entalpia h2
            T2_K = CP.PropsSI('T', 'P', P2_Pa, 'H', h2 * 1000.0, fluido)
            T2 = T2_K - T_kelvin_offset # T2 em °C
        except Exception as e:
            print(f"\nERRO ao calcular temperatura do estado 2 (Saída Real): {e}")
            return
        
        # Atribuição para o resultado final
        P_final_result = P_final
        T_final_result = T2
        h_final_result = h2
        
        P_inicial_result = P1
        T_inicial_result = T1
        h_inicial_result = h1
        
    else: # estado_conhecido == '2'
        # Conversão para unidades do CoolProp: K e Pa
        T2_K = T2 + T_kelvin_offset
        P2_Pa = P2 * P_pascal_to_kPa
        P1_Pa = P_final * P_pascal_to_kPa
        
        # 3.1 Propriedades da Saída (2)
        try:
            h2 = CP.PropsSI('H', 'P', P2_Pa, 'T', T2_K, fluido) / 1000.0  # h2 em kJ/kg
            s2 = CP.PropsSI('S', 'P', P2_Pa, 'T', T2_K, fluido) / 1000.0  # s2 em kJ/(kg*K)
        except Exception as e:
            print(f"\nERRO ao calcular propriedades do estado 2 (Saída): {e}")
            return
            
        # CÁLCULO REVERSO:
        # Assumindo água sub-resfriada (líquido) e incompressível para o trabalho isoentrópico
        
        # O volume específico (v) é aproximadamente 1/densidade (rho).
        try:
            rho_2 = CP.PropsSI('D', 'P', P2_Pa, 'T', T2_K, fluido)
            v_avg = 1.0 / rho_2 # m³/kg (Volume específico médio)
        except Exception as e:
            print(f"\nERRO ao calcular densidade: {e}")
            return
        
        P_diff = P2_Pa - P1_Pa # Pa
        
        # w_is (em J/kg) -> w_is / 1000.0 (em kJ/kg)
        w_is_kj = (v_avg * P_diff) / 1000.0 
        
        # Trabalho real w_real = w_is / Eta_is (em kJ/kg)
        w_real_kj = w_is_kj / Eta_is
        
        # w_real = h2 - h1 => h1 = h2 - w_real (em kJ/kg)
        h1 = h2 - w_real_kj
        
        # 3.2 Propriedades da Entrada Real (1) - Cálculo da Temperatura T1 e Entropia s1
        try:
            # T1 é a temperatura na pressão P1 e entalpia h1
            T1_K = CP.PropsSI('T', 'P', P1_Pa, 'H', h1 * 1000.0, fluido)
            T1 = T1_K - T_kelvin_offset # T1 em °C
            s1 = CP.PropsSI('S', 'P', P1_Pa, 'H', h1 * 1000.0, fluido) / 1000.0 # s1 em kJ/(kg*K)
        except Exception as e:
            print(f"\nERRO ao calcular propriedades do estado 1 (Entrada Real): {e}")
            return
            
        # Atribuição para o resultado final
        P_final_result = P_final
        T_final_result = T1
        h_final_result = h1
        
        P_inicial_result = P2
        T_inicial_result = T2
        h_inicial_result = h2
        Trabalho_is = w_is_kj
        Trabalho_real = w_real_kj

    # --- 4. Resultados Finais ---
    
    # 4.1. Potência do Fluido (Potência real baseada em Delta h e Eta_is)
    Potencia_do_Fluido = Vazao_m * Trabalho_real 
    
    # 4.2. Potência de Eixo (Incluindo a Eficiência Mecânica)
    # Potência de Eixo = Potência do Fluido / Eficiência Mecânica
    Potencia_do_Eixo = Potencia_do_Fluido / Eta_m

    # 4.3. Eficiência Total
    Eta_total = Eta_is * Eta_m
    
    # --- 5. Exibição ---
    
    print("\n" + "="*50)
    print("RESULTADOS DO MODELO DE BOMBA")
    print("="*50)
    
    # 5.1. Dados de Entrada (Atualizado para apresentar todas as entradas)
    print("DADOS DE ENTRADA UTILIZADOS:")
    if modo == '1':
        print(f"Modo: {EXEMPLO_ENTRADA['nome']}")
    elif modo == '2':
        print(f"Modo: {EXEMPLO_SAIDA['nome']}")
    else:
        print("Modo: Entrada Manual")
        
    print(f"Estado Conhecido: {'Entrada (1)' if estado_conhecido == '1' else 'Saída (2)'}")
    print(f"Vazão Mássica (ṁ): {Vazao_m:.3f} kg/s")
    print(f"Eficiência Isoentrópica (η_is): {Eta_is:.2f} ({(Eta_is * 100.0):.1f}%)")
    print(f"Eficiência Mecânica (η_m): {Eta_m:.2f} ({(Eta_m * 100.0):.1f}%)")
    print("-" * 50)
    
    # 5.2. Propriedades Calculadas
    
    if estado_conhecido == '1':
        print("\n--- Estado de Entrada (1) [Conhecido] ---")
        print(f"P1: {P_inicial_result:.2f} kPa")
        print(f"T1: {T_inicial_result:.2f} °C")
        print(f"h1: {h_inicial_result:.2f} kJ/kg")
        print(f"s1: {s1:.4f} kJ/(kg·K)")
        
        print("\n--- Estado de Saída Isoentrópica (2s) ---")
        print(f"P2s: {P_final_result:.2f} kPa (Pressão de Saída)")
        print(f"h2s: {h2s:.2f} kJ/kg")
        print(f"s2s: {s1:.4f} kJ/(kg·K) [Isoentrópica]")

        print("\n--- Estado de Saída Real (2) [Calculado] ---")
        print(f"P2: {P_final_result:.2f} kPa")
        print(f"T2: {T_final_result:.2f} °C")
        print(f"h2: {h_final_result:.2f} kJ/kg")
        
    else: # estado_conhecido == '2'
        print("\n--- Estado de Saída (2) [Conhecido] ---")
        print(f"P2: {P_inicial_result:.2f} kPa")
        print(f"T2: {T_inicial_result:.2f} °C")
        print(f"h2: {h_inicial_result:.2f} kJ/kg")
        print(f"s2: {s2:.4f} kJ/(kg·K)")
        
        # O modelo reverso usa a aproximação de líquido incompressível para w_is
        
        print("\n--- Estado de Entrada Real (1) [Calculado] ---")
        print(f"P1: {P_final_result:.2f} kPa")
        print(f"T1: {T_final_result:.2f} °C")
        print(f"h1: {h_final_result:.2f} kJ/kg")
        print(f"s1: {s1:.4f} kJ/(kg·K)")
        
    print("-" * 50)
    
    # 5.3. Trabalho e Potência
    print("TRABALHO E POTÊNCIA")
    print(f"Trabalho Isoentrópico (w_is): {Trabalho_is:.2f} kJ/kg")
    print(f"Trabalho Real (w_real): {Trabalho_real:.2f} kJ/kg")
    print("-" * 25)
    print(f"Potência Transferida ao Fluido (Ẇ_fluido): {Potencia_do_Fluido:.3f} kW")
    print(f"Potência de Eixo Requerida (Ẇ_eixo): {Potencia_do_Eixo:.3f} kW")
    print(f"Eficiência Total da Bomba (η_total): {Eta_total:.2f} ({(Eta_total * 100.0):.1f}%)")
    
    print("="*50)

# Executa a função principal
if __name__ == "__main__":
    calcular_propriedades_bomba()