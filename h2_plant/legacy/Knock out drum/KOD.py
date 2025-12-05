import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd

# --- CONSTANTES DE PROCESSO ---
# Temperatura e Pressão de entrada (SI Units)
T_IN_C = 4.0  # Temperatura de entrada (4 C)
T_IN_K = T_IN_C + 273.15  # Temperatura em Kelvin
P_IN_BAR = 40.0  # Pressão de entrada (40 bar)
P_IN_PA = P_IN_BAR * 1e5  # Pressão em Pascal (Pa)

# CONSTANTE DE CORREÇÃO (Valor universal dos gases)
R_UNIV = 8.31446  # J/(mol*K)

# Propriedades da Água Líquida (Simplificadas a 4ºC e 40 bar)
RHO_L_WATER = 1000.0  # kg/m³
# Pressão de saturação da água a 4°C (CoolProp)
P_SAT_H2O_PA = CP.PropsSI('P', 'T', T_IN_K, 'Q', 0, 'Water')

# Fator de Souders-Brown (K) para Demister Pad (Valor típico conservador: 0.08 m/s)
K_SOUDERS_BROWN = 0.08  # m/s

def modelar_knock_out_drum(gas_fluido: str, vazao_molar_in: float, delta_p_bar: float = 0.05, diametro_vaso_m: float = 1.0) -> dict:
    """
    Modelagem de um Knock-Out Drum (KOD) para fluxos de Hidrogênio ou Oxigênio.
    """
    
    # 1. CÁLCULO DAS PROPRIEDADES DE SAÍDA DO GÁS
    
    P_OUT_BAR = P_IN_BAR - delta_p_bar
    P_OUT_PA = P_OUT_BAR * 1e5
    
    try:
        y_H2O_out = P_SAT_H2O_PA / P_OUT_PA
        y_gas_out = 1.0 - y_H2O_out
        
        M_H2O = CP.PropsSI('M', 'Water')
        M_GAS = CP.PropsSI('M', gas_fluido)
        
        # CORREÇÃO DA FÓRMULA DE M_MIX_G e ESCOPO
        # M_MIX_G = y_gas_out * M_GAS + y_H2O_out * M_H2O  <-- CÁLCULO CORRETO
        M_MIX_G = y_gas_out * M_GAS + y_H2O_out * M_H2O
        
        Z_gas = CP.PropsSI('Z', 'T', T_IN_K, 'P', P_OUT_PA, gas_fluido)
        
        # Cálculo da densidade da mistura (Equação de Estado para gás real)
        rho_G_out = P_OUT_PA * M_MIX_G / (Z_gas * R_UNIV * T_IN_K)
        
        # Vazões Molares de Saída (Movidas para dentro do try)
        vazao_molar_gas_out = vazao_molar_in * y_gas_out
        vazao_molar_H2O_out = vazao_molar_in * y_H2O_out # Não é usada para o KOD, mas é útil na saída
        
        # Vazão Volumétrica de Saída (Movida para dentro do try)
        vazao_volumetrica_gas_out = vazao_molar_gas_out * M_MIX_G / rho_G_out
        
    except Exception as e:
        return {"ERRO": f"Erro no cálculo de propriedades do CoolProp para {gas_fluido}: {e}"}

    
    # 2. CÁLCULO DE DIMENSIONAMENTO (Agora usa variáveis definidas no try)
    
    V_max = K_SOUDERS_BROWN * np.sqrt((RHO_L_WATER - rho_G_out) / rho_G_out)
    
    A_min_requerida = vazao_volumetrica_gas_out / V_max
    
    A_vaso = np.pi * (diametro_vaso_m / 2)**2
    
    V_superficial_real = vazao_volumetrica_gas_out / A_vaso
    
    # 3. CÁLCULO ENERGÉTICO
    
    potencia_eletrica_adicional_W = vazao_volumetrica_gas_out * (delta_p_bar * 1e5)
    
    # --- RESULTADOS (Chaves ASCII) ---
    resultados = {
        # Para Tabela 1 (Entrada)
        "Temperatura (T_in)": [T_IN_C, "°C"],
        "Pressao (P_in)": [P_IN_BAR, "bar"],
        "Vazao Molar Total": [vazao_molar_in, "mol/s"],
        "Diametro do Vaso": [diametro_vaso_m, "m"],
        "Perda de Pressao (Delta P)": [delta_p_bar, "bar"],
        
        # Para Tabela 2 (Saída)
        "Pressao de Saida (P_out)": [P_OUT_BAR, "bar"],
        "Temperatura de Saida (T_out)": [T_IN_C, "°C"],
        "Fracao Molar H2O (y_H2O)": [y_H2O_out, "-"],
        "Vazao Molar Gas (F_Gas)": [vazao_molar_gas_out, "mol/s"],
        "Vazao Volumetrica (V_Gas)": [vazao_volumetrica_gas_out, "m^3/s"],
        "Densidade Media (rho_mix)": [rho_G_out, "kg/m^3"],
        "Consumo Elet. Adicional": [potencia_eletrica_adicional_W, "W"],
        
        # Para Tabela 3 (Dimensionamento)
        "Densidade do Gas (rho_G)": [rho_G_out, "kg/m^3"],
        "Velocidade Max. Permissivel (V_max)": [V_max, "m/s"],
        "Velocidade Superficial Real (V_real)": [V_superficial_real, "m/s"],
        "Area Minima Requerida": [A_min_requerida, "m^2"],
        "STATUS SEPARACAO": [("OK" if V_superficial_real < V_max else "ATENÇÃO: Vaso subdimensionado!"), "-"]
    }
    
    return resultados

def imprimir_resultados_em_tabelas_pandas(res_h2: dict, res_o2: dict):
    """
    Gera e imprime as três tabelas formatadas usando DataFrames do Pandas no formato transposto.
    """
    
    # --- Parâmetros que definem o agrupamento das tabelas (Usando chaves ASCII) ---
    params_entrada = ["Temperatura (T_in)", "Pressao (P_in)", "Vazao Molar Total", "Diametro do Vaso", "Perda de Pressao (Delta P)"]
    
    params_saida = ["Pressao de Saida (P_out)", "Temperatura de Saida (T_out)", "Fracao Molar H2O (y_H2O)", "Vazao Molar Gas (F_Gas)", "Vazao Volumetrica (V_Gas)", "Densidade Media (rho_mix)", "Consumo Elet. Adicional"]
    
    params_dimensionamento = ["Densidade do Gas (rho_G)", "Velocidade Max. Permissivel (V_max)", "Velocidade Superficial Real (V_real)", "Area Minima Requerida", "STATUS SEPARACAO"]

    def create_dataframe_transposed(res_h2: dict, res_o2: dict, param_list: list) -> pd.DataFrame:
        """Cria um DataFrame transposto e formata valores numéricos."""
        
        data_h2 = []
        data_o2 = []
        units = []
        
        for p in param_list:
            valor_h2, unidade = res_h2[p]
            valor_o2, _ = res_o2[p]
            units.append(unidade)
            
            # Lógica de formatação de precisão:
            if p == "Fracao Molar H2O (y_H2O)" and isinstance(valor_h2, (int, float)):
                data_h2.append(f"{valor_h2:.8f}")
                data_o2.append(f"{valor_o2:.8f}")
            elif isinstance(valor_h2, (int, float)):
                data_h2.append(f"{valor_h2:.4f}")
                data_o2.append(f"{valor_o2:.4f}")
            else:
                data_h2.append(valor_h2)
                data_o2.append(valor_o2)

        # 2. Criação do DataFrame
        data = {
            "H2": data_h2,
            "O2": data_o2,
            "Unidade": units
        }
        df = pd.DataFrame(data, index=param_list)
        df = df[['H2', 'O2', 'Unidade']]
        
        return df
    
    # --- Tabela 1: Informações de Entrada ---
    print("\n## ⚙️ 1. Tabela de Informações de Entrada (Transposta)")
    df_entrada = create_dataframe_transposed(res_h2, res_o2, params_entrada)
    print(df_entrada.to_markdown(numalign="left", stralign="left"))
    
    # --- Tabela 2: Resultados de Saída (Fluxo de Gás e Energia) ---
    print("\n## 📈 2. Tabela de Resultados de Saída (Fluxo de Gás e Energia) (Transposta)")
    df_saida = create_dataframe_transposed(res_h2, res_o2, params_saida)
    print(df_saida.to_markdown(numalign="left", stralign="left"))
    
    # --- Tabela 3: Dimensionamento do KOD ---
    print("\n## 📐 3. Tabela de Dimensionamento do KOD (Transposta)")
    df_dimensionamento = create_dataframe_transposed(res_h2, res_o2, params_dimensionamento)
    print(df_dimensionamento.to_markdown(numalign="left", stralign="left"))
    
# --- EXEMPLO DE USO ---

# Para fins de exemplo (ajuste estes valores conforme sua necessidade):
vazao_H2_exemplo = 100.0  # mol/s
vazao_O2_exemplo = 50.0   # mol/s 

# 1. Executar o modelo para H2 e O2
resultado_H2 = modelar_knock_out_drum('H2', vazao_H2_exemplo)
resultado_O2 = modelar_knock_out_drum('O2', vazao_O2_exemplo)

# 2. Imprimir os resultados no formato de tabela (usando Pandas)
if "ERRO" not in resultado_H2 and "ERRO" not in resultado_O2:
    imprimir_resultados_em_tabelas_pandas(resultado_H2, resultado_O2)
else:
    print("Ocorreu um erro durante a modelagem de um dos fluidos.")
    print(f"Erro H2: {resultado_H2.get('ERRO', 'Nenhum')}")
    print(f"Erro O2: {resultado_O2.get('ERRO', 'Nenhum')}")