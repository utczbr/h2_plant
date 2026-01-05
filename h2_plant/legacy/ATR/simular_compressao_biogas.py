import pandas as pd
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from modelo_compressor import modelo_compressor_ideal
from modelo_dry_cooler import simular_dry_cooler_com_setpoint

# --- PARÂMETROS DO SISTEMA (RECIPROCANTE) ---
P_IN_BAR, T_IN_C, P_FINAL_BAR = 3.0, 25.0, 15.0
T_LIMIT_C = 120.0        
T_DC_TARGET = 40.0       
ETA_IS = 0.75      
ETA_M = 0.95       

def calc_P_out_analitico(T_in, P_in, T_target, eta=0.75):
    T1, T2, K_EXP = T_in + 273.15, T_target + 273.15, 0.285
    return P_in * ((eta * (T2/T1 - 1) + 1)**(1/K_EXP))

def calc_T_in_analitico(P_in, P_out, T_target, eta=0.75):
    T2, pr, K_EXP = T_target + 273.15, P_out / P_in, 0.285
    return (T2 / (1 + (1/eta) * (pr**K_EXP - 1))) - 273.15

def processar_linha(vazao_h):
    m_dot_s = vazao_h / 3600.0
    
    # --- ESTÁGIO 1 ---
    P1_bar = min(calc_P_out_analitico(T_IN_C, P_IN_BAR, T_LIMIT_C, eta=ETA_IS), P_FINAL_BAR)
    c1 = modelo_compressor_ideal(T_IN_C, P_IN_BAR*1e5, P1_bar*1e5, m_dot_s, Eta_is=ETA_IS, Eta_m=ETA_M)
    T_dc1, W_dc1 = simular_dry_cooler_com_setpoint(c1['T_out_C'], T_DC_TARGET, m_dot_s)
    
    # --- ESTÁGIO 2 ---
    P2_calc = calc_P_out_analitico(T_dc1, P1_bar, T_LIMIT_C, eta=ETA_IS)
    P2_bar = max(P1_bar + 0.1, min(P2_calc, P_FINAL_BAR - 0.5))
    c2 = modelo_compressor_ideal(T_dc1, P1_bar*1e5, P2_bar*1e5, m_dot_s, Eta_is=ETA_IS, Eta_m=ETA_M)
    
    # Resfriador 2 Otimizado
    T_needed_in3 = calc_T_in_analitico(P2_bar, P_FINAL_BAR, T_LIMIT_C, eta=ETA_IS)
    T_set2 = max(T_DC_TARGET, min(T_needed_in3, c2['T_out_C']))
    T_dc2, W_dc2 = simular_dry_cooler_com_setpoint(c2['T_out_C'], T_set2, m_dot_s)
    
    # --- ESTÁGIO 3 ---
    c3 = modelo_compressor_ideal(T_dc2, P2_bar*1e5, P_FINAL_BAR*1e5, m_dot_s, Eta_is=ETA_IS, Eta_m=ETA_M)
    
    # Retorno com as 17 colunas originais
    return [
        vazao_h, P_IN_BAR, T_IN_C,                       # 1, 2, 3
        c1['P_out_bar'], c1['T_out_C'], c1['W_dot_comp_W']/1000, # 4, 5, 6
        T_dc1, W_dc1/1000,                               # 7, 8
        c2['P_out_bar'], c2['T_out_C'], c2['W_dot_comp_W']/1000, # 9, 10, 11
        T_dc2, W_dc2/1000,                               # 12, 13
        c3['P_out_bar'], c3['T_out_C'], c3['W_dot_comp_W']/1000, # 14, 15, 16
        (c1['W_dot_comp_W'] + c2['W_dot_comp_W'] + c3['W_dot_comp_W'] + W_dc1 + W_dc2)/1000 # 17
    ]

def main():
    try:
        df = pd.read_csv("biogas.CSV", sep="|", decimal=",") 
    except:
        df = pd.read_csv("biogas.CSV", decimal=",")
    
    vazoes = df.iloc[:, 0].tolist()

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        resultados = list(executor.map(processar_linha, vazoes))

    colunas = [
        "vazao_kg_h", "P0_bar", "T0_C",
        "P1_bar", "T1_C", "W1_kW", "T_dc1_C", "W_dc1_kW",
        "P2_bar", "T2_C", "W2_kW", "T_dc2_C", "W_dc2_kW",
        "P3_bar", "T3_C", "W3_kW", "W_total_sistema_kW"
    ]
    
    df_res = pd.DataFrame(resultados, columns=colunas)
    output_filename = "resultado_compressao_sem_aquecedor.csv"
    df_res.to_csv(output_filename, index=False, sep=";", decimal=",")
    
    print(f"Arquivo gerado: {output_filename}")
    print(f"Consumo DC1 na vazão máxima: {df_res['W_dc1_kW'].max()*1000:.2f} W")

if __name__ == "__main__":
    main()