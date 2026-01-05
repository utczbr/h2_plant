import pandas as pd
import numpy as np
import CoolProp.CoolProp as CP

def calcular_entalpia_biogas(T_K, P_Pa):
    """
    Calcula a entalpia do biogás (60% CH4, 40% CO2 molar) como mistura ideal
    para garantir estabilidade numérica no CoolProp.
    """
    # Massas molares (g/mol)
    M_CH4, M_CO2 = 16.042, 44.01
    y_CH4, y_CO2 = 0.6, 0.4
    M_mix = y_CH4 * M_CH4 + y_CO2 * M_CO2
    
    # Frações mássicas
    w_CH4 = (y_CH4 * M_CH4) / M_mix
    w_CO2 = (y_CO2 * M_CO2) / M_mix
    
    # Entalpias individuais [J/kg]
    h_CH4 = CP.PropsSI('H', 'T', T_K, 'P', P_Pa, 'Methane')
    h_CO2 = CP.PropsSI('H', 'T', T_K, 'P', P_Pa, 'CarbonDioxide')
    
    return w_CH4 * h_CH4 + w_CO2 * h_CO2

def gerar_modelo_entrada_atr_completo():
    # --- 1. CARREGAMENTO DOS DADOS EXISTENTES ---
    try:
        df_reg = pd.read_csv('ATR_linear_regressions.csv', sep=';', decimal=',')
        df_comp = pd.read_csv('resultado_compressao_sem_aquecedor.csv', sep=';', decimal=',')
    except Exception as e:
        print(f"Erro ao carregar arquivos: {e}")
        return

    # --- 2. PARÂMETROS E PREMISSAS DO SISTEMA (Baseado na tese de Carlini) ---
    P_ATM_PA = 101325.0
    P_FINAL_BAR = 15.0
    P_FINAL_PA = P_FINAL_BAR * 1e5
    T_ALVO_MISTURA_C = 500.0
    T_ALVO_K = T_ALVO_MISTURA_C + 273.15
    
    # Condições de entrada consideradas
    T_REF_O2_C = 25.0              # Oxigênio da eletrólise/ambiente
    T_REF_AGUA_C = 25.0            # Água líquida da rede/tratamento
    ETA_BOMBA = 0.70               # Eficiência isentrópica da bomba d'água

    # --- 3. PRÉ-CÁLCULO DE PROPRIEDADES FIXAS ---
    h_bio_500 = calcular_entalpia_biogas(T_ALVO_K, P_FINAL_PA)
    h_o2_500 = CP.PropsSI('H', 'T', T_ALVO_K, 'P', P_FINAL_PA, 'Oxygen')
    h_water_500 = CP.PropsSI('H', 'T', T_ALVO_K, 'P', P_FINAL_PA, 'Water')
    
    h_o2_in = CP.PropsSI('H', 'T', T_REF_O2_C + 273.15, 'P', P_FINAL_PA, 'Oxygen')
    h_water_liq_in = CP.PropsSI('H', 'T', T_REF_AGUA_C + 273.15, 'P', P_ATM_PA, 'Water')

    resultados_finais = []

    print(f"Processando balanço de massa e energia para {len(df_reg)} pontos...")

    # --- 4. PROCESSAMENTO LINHA A LINHA ---
    for i in range(len(df_reg)):
        # Vazões do modelo linear
        f_o2_kmol = df_reg.loc[i, 'x']
        m_bio = df_reg.loc[i, 'Fm_bio_func']
        m_o2 = df_reg.loc[i, 'Fm_O2_func']
        m_steam_in = df_reg.loc[i, 'Fm_steam_func'] # Água de entrada para o vapor
        
        # Dados da compressão do biogás
        t_bio_in_c = df_comp.loc[i, 'T3_C']
        w_comp_total_kw = df_comp.loc[i, 'W_total_sistema_kW']
        
        # A) CÁLCULO DA BOMBA DE ÁGUA (Líquido: 1 bar -> 15 bar)
        v_agua = 1 / CP.PropsSI('D', 'T', T_REF_AGUA_C + 273.15, 'P', P_ATM_PA, 'Water')
        # Trabalho [W] = m_dot(kg/s) * v * deltaP / eta
        w_bomba_w = (m_steam_in / 3600) * v_agua * (P_FINAL_PA - P_ATM_PA) / ETA_BOMBA
        h_water_pos_bomba = h_water_liq_in + (w_bomba_w / (m_steam_in / 3600))

        # B) BALANÇO NO MISTURADOR (Mixer)
        h_bio_in = calcular_entalpia_biogas(t_bio_in_c + 273.15, P_FINAL_PA)
        
        # O vapor deve compensar o aquecimento do Biogás e O2 até 500°C
        delta_H_bio = m_bio * (h_bio_500 - h_bio_in)
        delta_H_o2 = m_o2 * (h_o2_500 - h_o2_in)
        h_vapor_req = h_water_500 + (delta_H_bio + delta_H_o2) / m_steam_in
        
        try:
            t_vapor_req_c = CP.PropsSI('T', 'H', h_vapor_req, 'P', P_FINAL_PA, 'Water') - 273.15
        except:
            t_vapor_req_c = np.nan

        # C) POTÊNCIA DO AQUECEDOR (Aquecimento + Vaporização + Superaquecimento)
        q_heater_kw = m_steam_in * (h_vapor_req - h_water_pos_bomba) / (3600 * 1000)

        # --- 5. CONSOLIDAÇÃO DOS DADOS NO DICIONÁRIO ---
        resultados_finais.append({
            # Identificação
            'F_O2_entrada_kmol_h': f_o2_kmol,
            
            # Vazões de Massa (Entradas)
            'm_biogas_entrada_kg_h': m_bio,
            'm_oxigenio_entrada_kg_h': m_o2,
            'm_agua_vapor_entrada_kg_h': m_steam_in,
            
            # Condições de Entrada Consideradas
            'P_entrada_biogas_bar': P_FINAL_BAR,
            'T_entrada_biogas_C': t_bio_in_c,
            'P_entrada_oxigenio_bar': P_FINAL_BAR,
            'T_entrada_oxigenio_C': T_REF_O2_C,
            'P_entrada_agua_bar': 1.0,
            'T_entrada_agua_C': T_REF_AGUA_C,
            
            # Resultados do Vapor e Aquecimento
            'T_vapor_necessaria_C': t_vapor_req_c,
            'Q_aquecedor_vapor_kW': q_heater_kw,
            'W_bomba_agua_W': w_bomba_w,
            
            # Resultados da Mistura Final
            'P_mistura_final_bar': P_FINAL_BAR,
            'T_mistura_final_alvo_C': T_ALVO_MISTURA_C,
            
            # Consumo de Energia Anterior (Compressão Biogás)
            'W_compressao_biogas_total_kW': w_comp_total_kw
        })

    # --- 6. EXPORTAÇÃO PARA EXCEL (CSV) ---
    final_df = pd.DataFrame(resultados_finais)
    output_file = 'resultado_completo_entrada_atr.csv'
    final_df.to_csv(output_file, sep=';', decimal=',', index=False)
    print(f"Sucesso! Relatório consolidado gerado em: {output_file}")

if __name__ == "__main__":
    gerar_modelo_entrada_atr_completo()