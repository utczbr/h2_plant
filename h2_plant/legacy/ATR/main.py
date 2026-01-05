# main.py
import pandas as pd
import numpy as np
import traceback
from modelo_dry_cooler import modelar_resfriador_condensador
from modelo_ciclone import modelar_ciclone_separador
from modelo_chiller import modelar_chiller_gas
from modelo_coalescedor import modelar_coalescedor
from modelo_psa import modelar_psa
from modelo_trocador_hex import modelar_hex_recuperacao
from modelo_compressor import modelo_compressor_ideal
from modelo_mixer import mixer_model
from modelo_boiler import simular_boiler_fluxo_continuo

def executar_simulacao():
    P_OP = 15.0
    T_CHILLER = 4.0
    COP_ESTIMADO = 2.5 
    T_W_REPO = 25.0
    T_BOILER_TARGET = 30.0 
    
    P_FINAL_TARGET = 40.0
    T_INTERCOOLER_TARGET = 30.0
    P_INTERMEDIARIA = 28.0 
    
    T_W_IN_HEX = 25.0              
    # T_W_TARGET_HEX removido: agora é calculado pela física do trocador (Area fixa)
    M_DOT_W_HEX = 4224 / 3600 # 1.173 kg/s

    def get_gas_cp_validated(row):
        xH2 = row['xH2_offgas_func']
        xCO2 = row['xCO2_offgas_func']
        xCH4 = row['xCH4_offgas_func']
        xResto = 1.0 - (xH2 + xCO2 + xCH4)
        return (xH2 * 14.5 + xCO2 * 1.1 + xCH4 * 3.0 + xResto * 2.1)

    try:
        df = pd.read_csv('ATR_linear_regressions.csv', sep=';', decimal=',')
        results = []

        for _, row in df.iterrows():
            m_dot_gas = row['Fm_offgas_func'] / 3600.0
            m_dot_h2o_tot = row['Fm_water_func'] / 3600.0
            m_dot_steam_target = row['Fm_steam_func'] / 3600.0
            cp_mix = get_gas_cp_validated(row)
            
            # --- 1. RESFRIADORES E HEX (Área Fixa 152m2) ---
            h08 = modelar_resfriador_condensador(m_dot_gas, m_dot_h2o_tot, row['Tin_H08_func'], row['Tout_H08_func'], P_OP, cp_mix)
            h09 = modelar_resfriador_condensador(m_dot_gas, m_dot_h2o_tot, row['Tin_H09_func'], row['Tout_H09_func'], P_OP, cp_mix)
            
            # Chamada atualizada com remoção do target de temperatura fixo
            hex_res = modelar_hex_recuperacao(m_dot_gas, cp_mix, row['Tin_H05_func'], T_W_IN_HEX, M_DOT_W_HEX)
            t_saida_hex = hex_res['T_out_gas']
            t_saida_agua_hex = hex_res['T_out_w']
            
            # --- 2. BALANÇO DE ÁGUA E CICLONES ---
            cond_hex = modelar_resfriador_condensador(m_dot_gas, m_dot_h2o_tot, row['Tin_H05_func'], t_saida_hex, P_OP, cp_mix)
            cic1 = modelar_ciclone_separador('Water', cond_hex['m_dot_h2o_liq_no_fluxo'], P_OP, t_saida_hex)
            m_h2o_pos_cic1 = m_dot_h2o_tot - cic1['m_dot_agua_removida_kg_s']
            
            chil_phys = modelar_resfriador_condensador(m_dot_gas, m_h2o_pos_cic1, t_saida_hex, T_CHILLER, P_OP, cp_mix)
            cic2 = modelar_ciclone_separador('Water', chil_phys['m_dot_h2o_liq_no_fluxo'], P_OP, T_CHILLER)
            m_h2o_pos_cic2 = m_h2o_pos_cic1 - cic2['m_dot_agua_removida_kg_s']
            
            # --- 3. MISTURADORES ---
            fluxos_ciclones = [{'m_dot': cic1['m_dot_agua_removida_kg_s'], 'T': t_saida_hex, 'P': P_OP},
                               {'m_dot': cic2['m_dot_agua_removida_kg_s'], 'T': T_CHILLER, 'P': P_OP}]
            mix_recup = mixer_model(fluxos_ciclones, P_OP)
            m_dot_recuperada = mix_recup['m_dot_out_kg_s']
            m_dot_makeup = max(0, m_dot_steam_target - m_dot_recuperada)
            fluxos_finais = [{'m_dot': m_dot_recuperada, 'T': mix_recup['T_out_C'], 'P': P_OP},
                             {'m_dot': m_dot_makeup, 'T': T_W_REPO, 'P': P_OP}]
            mix_final = mixer_model(fluxos_finais, P_OP)
            
            # --- 4. TRATAMENTO DE GÁS, BOILER E COMPRESSÃO ---
            m_liq_para_coal = chil_phys['m_dot_h2o_liq_no_fluxo'] - cic2['m_dot_agua_removida_kg_s']
            coal = modelar_coalescedor(m_liq_para_coal)
            
            res_boiler = simular_boiler_fluxo_continuo(m_dot_gas, cp_mix, T_CHILLER, T_BOILER_TARGET)
            
            m_h2o_para_psa = m_h2o_pos_cic2 - coal['m_dot_agua_removida_kg_s']
            psa = modelar_psa(m_h2o_para_psa, P_OP)
            
            p_in_c1 = psa['P_out_bar']
            c1 = modelo_compressor_ideal('hydrogen', T_BOILER_TARGET, p_in_c1*1e5, P_INTERMEDIARIA*1e5, m_dot_gas, m_dot_gas)
            c2 = modelo_compressor_ideal('hydrogen', T_INTERCOOLER_TARGET, P_INTERMEDIARIA*1e5, P_FINAL_TARGET*1e5, m_dot_gas, m_dot_gas)
            
            # --- 5. CONSOLIDAÇÃO ---
            w_chiller_el = chil_phys['Q_total_kW'] / COP_ESTIMADO
            w_comp1_kw = c1['W_dot_comp_W'] / 1000.0
            w_comp2_kw = c2['W_dot_comp_W'] / 1000.0
            w_total = w_chiller_el + coal['W_dot_kW'] + psa['W_dot_kW'] + w_comp1_kw + w_comp2_kw + res_boiler['W_dot_kW']
            ppm_final = ((m_h2o_para_psa - psa['m_dot_agua_removida_kg_s']) / m_dot_gas) * 1_000_000

            results.append({
                'F_O2': row['x'],
                'T_out_H08': row['Tout_H08_func'], 'Q_H08': h08['Q_total_kW'], 'Q_H08_plan': abs(row['H08_Q_func']),
                'T_out_H09': row['Tout_H09_func'], 'Q_H09': h09['Q_total_kW'], 'Q_H09_plan': abs(row['H09_Q_func']),
                'T_in_HEX': row['Tin_H05_func'], 'T_out_HEX': t_saida_hex, 'Q_HEX': hex_res['Q_hex_kW'],
                'T_in_H2O_HEX': T_W_IN_HEX, 'F_H2O_HEX': M_DOT_W_HEX, 'T_out_H2O_HEX': t_saida_agua_hex,
                'T_out_Chil': T_CHILLER, 'Q_Chil': chil_phys['Q_total_kW'],
                'H2O_rem_Cic1': cic1['m_dot_agua_removida_kg_h'],
                'H2O_rem_Cic2': cic2['m_dot_agua_removida_kg_h'],
                'H2O_rem_Coal': coal['m_dot_agua_removida_kg_s'] * 3600,
                'H2O_rem_PSA': psa['m_dot_agua_removida_kg_s'] * 3600,
                'W_Chil': w_chiller_el, 'W_Coal': coal['W_dot_kW'], 'W_PSA': psa['W_dot_kW'], 'W_Boiler': res_boiler['W_dot_kW'],
                'W_Comp1': w_comp1_kw, 'W_Comp2': w_comp2_kw, 'W_Total': w_total,
                'P_PSA_Out': p_in_c1, 'T_C1_Out': c1['T_C'], 'P_Inter': P_INTERMEDIARIA, 
                'T_IC_Out': T_INTERCOOLER_TARGET, 'T_C2_Out': c2['T_C'], 
                'P_Fin': P_FINAL_TARGET, 'H2O_PPM_Final': ppm_final,
                'F_H2O_Recup': mix_recup['m_dot_out_kg_h'], 'T_H2O_Recup': mix_recup['T_out_C'],
                'F_H2O_Makeup': m_dot_makeup * 3600, 'F_H2O_ATR_Final': mix_final['m_dot_out_kg_h'],
                'T_H2O_ATR_Final': mix_final['T_out_C'], 'F_H2O_Target_Plan': row['Fm_steam_func'],
                'T_Boiler_In': T_CHILLER, 'T_Boiler_Out': res_boiler['T_out_C']
            })
        return pd.DataFrame(results)
    except Exception as e:
        print(f"Erro na execução do modelo: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    from gerador_tabelas import imprimir_tabelas
    from gerador_graficos import gerar_graficos
    df_res = executar_simulacao()
    if df_res is not None:
        imprimir_tabelas(df_res)
        gerar_graficos(df_res)