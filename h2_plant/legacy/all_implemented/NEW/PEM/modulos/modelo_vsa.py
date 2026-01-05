# modelo_vsa_EDO.py
import numpy as np
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
import sys

# Fator de Compressibilidade Z de H2 puro a 40 bar e ~15°C (Z ≈ 1.050)
Z_H2_FIXO = 1.050 

def modelo_vsa_dimensionamento_parcial(
    T_entrada_C,
    P_entrada_bar,
    vazao_m3_h,
    umidade_molar_entrada_ppm,
    P_adsorcao_bar,
    P_produto_bar,
    P_regeneracao_bar,
    eficiencia_compressor=0.75,
    eficiencia_bomba_vacuo=0.60,
    recuperacao_h2=0.90,
    umidade_molar_max_ppm=50000,
    tempo_ciclo_total_min=10.0, # Duração de um ciclo completo (min)
    capacidade_trabalho_h2o_kg_kg=0.03, # Capacidade de trabalho do sorvente (kg H2O / kg sorvente)
    n_leitos=3 # Número de leitos (mínimo de 2 para operação contínua)
):
    """
    Simula o estado, estima o consumo energético e a massa de adsorvente do VSA.
    Adapta a saída para ser compatível com o sistema de simulação central.
    """

    # --- 0. Verificação do Limite de Umidade ---
    if umidade_molar_entrada_ppm > umidade_molar_max_ppm:
        return {
            "erro": f"A umidade de entrada ({umidade_molar_entrada_ppm:.0f} ppm) excede o limite VSA ({umidade_molar_max_ppm:.0f} ppm).",
            "T_C": T_entrada_C,
            "P_bar": P_entrada_bar,
            "Q_dot_fluxo_W": 0.0,
            "W_dot_comp_W": 0.0,
            "Agua_Removida_kg_s": 0.0,
            "y_H2O_out": umidade_molar_entrada_ppm / 1e6
        }

    # --- 1. Definição das Variáveis de Entrada e Constantes ---
    T_entrada_K = T_entrada_C + 273.15
    P_entrada_Pa = P_entrada_bar * 1e5
    P_adsorcao_Pa = P_adsorcao_bar * 1e5
    P_produto_Pa = P_produto_bar * 1e5
    P_regeneracao_Pa = P_regeneracao_bar * 1e5
    
    # Frações molares
    x_h2o_in = umidade_molar_entrada_ppm / 1e6
    x_h2_in = 1.0 - x_h2o_in
    M_H2 = PropsSI('MOLAR_MASS', 'H2') 
    M_H2O = PropsSI('MOLAR_MASS', 'H2O')
    
    # Massa Molar Média da Mistura
    M_mistura = x_h2_in * M_H2 + x_h2o_in * M_H2O
    w_h2_in = x_h2_in * M_H2 / M_mistura
    w_h2o_in = x_h2o_in * M_H2O / M_mistura
    
    # --- 2. Cálculo do Fluxo de Massa de Entrada (USANDO GÁS REAL COM FATOR Z FIXO) ---
    try:
        # R_UNIV é a constante universal dos gases (J/(mol*K))
        R_UNIV = 8.31446 # Valor numérico fixo

        # Densidade Mássica (rho = P * M_mistura / (Z * R_UNIV * T))
        # O fator Z é aplicado aqui para corrigir a densidade (Z_H2_FIXO = 1.050)
        rho_in = (P_entrada_Pa * M_mistura) / (Z_H2_FIXO * R_UNIV * T_entrada_K)
        
        m_dot_in_kg_s = vazao_m3_h * rho_in / 3600 # Vazão mássica total de entrada
        m_dot_h2_in = m_dot_in_kg_s * w_h2_in # Vazão mássica de H2 principal
        
    except Exception as e:
        return {"erro": f"Erro no cálculo da densidade com Fator Z: {e}", "consumo_energetico_total_kW": 0.0}

    # --- 3. Simulação da Compressão (Energia de Adsorção) ---
    try:
        # Propriedades usando H2 PURO (mais estável para W dot)
        # NOTA: O cálculo de W_dot deve idealmente usar propriedades da mistura, 
        # mas CoolProp pode ser instável para misturas complexas. Mantemos H2 puro para estabilidade, 
        # aplicando o W_dot na massa TOTAL.
        
        # Pressão média (adsorção + produto/entrada) para cálculo de entalpia
        P_ref_Pa = (P_adsorcao_Pa + P_entrada_Pa) / 2.0
        h_in = PropsSI('H', 'T', T_entrada_K, 'P', P_ref_Pa, 'H2')
        s_in = PropsSI('S', 'T', T_entrada_K, 'P', P_ref_Pa, 'H2')
        
        # Compressão (Adsorção) - Mantida para P_adsorcao
        h_adsorcao_s = PropsSI('H', 'P', P_adsorcao_Pa, 'S', s_in, 'H2')
        
        # O trabalho é calculado sobre a massa da mistura total (m_dot_in_kg_s)
        trabalho_compressor_real_por_massa = (h_adsorcao_s - h_in) / eficiencia_compressor
        P_compressao_kW = m_dot_in_kg_s * trabalho_compressor_real_por_massa / 1000.0

        # --- 4. Simulação do Vácuo (Energia de Regeneração) ---
        m_dot_h2_produto = m_dot_h2_in * recuperacao_h2
        # A purga inclui o H2 perdido (m_dot_in_kg_s * (1 - recuperacao_h2)) e o vapor de água removido.
        # Simplificamos assumindo que o fluxo de purga é uma proporção da massa total de entrada.
        m_dot_purga = m_dot_in_kg_s * (1.0 - recuperacao_h2) + m_dot_in_kg_s * w_h2o_in # Massa perdida + massa de água
        P_descarte_Pa = 1.01325e5 # 1 atm
        
        # Propriedades usando H2 PURO (aproximação para o gás de purga/regeneração)
        h_reg_in = PropsSI('H', 'T', T_entrada_K, 'P', P_regeneracao_Pa, 'H2')
        s_reg_in = PropsSI('S', 'T', T_entrada_K, 'P', P_regeneracao_Pa, 'H2')
        h_reg_out_s = PropsSI('H', 'P', P_descarte_Pa, 'S', s_reg_in, 'H2')
        
        trabalho_bomba_real_por_massa = (h_reg_out_s - h_reg_in) / eficiencia_bomba_vacuo
        # O trabalho é negativo para a bomba de vácuo (consumo). Usamos abs() para consumo.
        P_vacuo_kW = m_dot_purga * trabalho_bomba_real_por_massa / 1000.0
    
    except Exception as e:
         # Este bloco não deve falhar se o CoolProp estiver estável para H2 puro
         return {"erro": f"Erro no cálculo termodinâmico (entalpia/entropia) com H2 puro: {e}", "consumo_energetico_total_kW": 0.0}

    # --- 5. Dimensionamento Parcial (Massa de Adsorvente) ---
    m_dot_h2o_in = m_dot_in_kg_s * w_h2o_in # kg H2O/s
    
    tempo_adsorcao_s = (tempo_ciclo_total_min * 60) / n_leitos 
    
    m_agua_removida_por_leito_kg = m_dot_h2o_in * tempo_adsorcao_s
    
    m_ads_por_leito_kg = m_agua_removida_por_leito_kg / capacidade_trabalho_h2o_kg_kg
    
    m_ads_total_kg = m_ads_por_leito_kg * n_leitos

    # --- 6. Resultados Finais e Energia Específica ---
    # Potência Total: Compressão + Vácuo (assumimos consumo elétrico como positivo)
    P_total_kW = P_compressao_kW + abs(P_vacuo_kW) 
    
    Y_H2O_OUT_ALVO = 100e-6 # 100 ppm molar (Para ser consistente com LIMITES['PSA']['y_H2O_MAX_PPM'])
    
    estado_produto_h2 = {
        "temperatura_C": T_entrada_C, # Assumindo T_out ≈ T_in para o produto VSA
        "pressao_bar": P_produto_bar,
        "vazao_massica_produto_kg_s": m_dot_h2_produto
    }
    
    # --- 7. Mapeamento para a Saída do Sistema Central ---
    return {
        # Saídas Mínimas para o sistema central
        "T_C": estado_produto_h2["temperatura_C"],
        "P_bar": estado_produto_h2["pressao_bar"],
        "Q_dot_fluxo_W": 0.0, 
        "W_dot_comp_W": P_total_kW * 1000, 
        "Agua_Removida_kg_s": m_dot_h2o_in, # Toda a água de entrada é removida (idealmente)
        "y_H2O_out": Y_H2O_OUT_ALVO, # Pureza alvo (molar)
        "m_dot_gas_out_princ": m_dot_h2_produto, # Vazão mássica de H2 produto (principal)
        
        # Dados de Dimensionamento e Custo para Resumo
        "dimensionamento_parcial": {
            "massa_adsorvente_total_kg": m_ads_total_kg,
            "massa_adsorvente_por_leito_kg": m_ads_por_leito_kg,
            "tempo_adsorcao_por_leito_s": tempo_adsorcao_s,
            "vazao_h2o_removida_kg_h": m_dot_h2o_in * 3600
        }
        ,
        "consumo_energetico": {
            "potencia_total_kW": P_total_kW,
            "energia_especifica_kwh_por_kg_h2": P_total_kW / (m_dot_h2_produto * 3600 / 1000.0)
        }
    }