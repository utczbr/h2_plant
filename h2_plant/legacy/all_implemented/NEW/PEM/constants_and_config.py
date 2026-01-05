# constants_and_config.py
# Centraliza todas as constantes e configurações do sistema de tratamento de gás.

import numpy as np
import CoolProp.CoolProp as CP
import sys

# =================================================================
# === CONFIGURAÇÕES GLOBAIS DO ELETROLISADOR PEM (5 MW) ===
# =================================================================
# Potência Nominal: 5.0 MW
P_NOMINAL_KW = 5000.0
# Consumo Específico de Energia (SEC)
SEC_KWH_KG_H2 = 56.18 
# Vazão de Água de Recirculação (Loop Anódico): 250,000 kg/h
M_DOT_H2O_RECIRC_TOTAL_KGS = 250000.0 / 3600.0 # 250000 kg/h (~69.4444 kg/s)
# Coeficiente de Drag Electro-osmótico (ndrag): 1.1 mol H2O / mol H+

# =================================================================
# === CONSTANTES DE BALANÇO DE ÁGUA (BASEADO NA LÓGICA DO USUÁRIO) ===
# =================================================================
# 1. CÁLCULO DA PRODUÇÃO E CONSUMO (Baseado em Constantes)
M_DOT_G_H2_KGS_PRODUZIDA = (P_NOMINAL_KW / SEC_KWH_KG_H2) / 3600.0
M_DOT_G_H2 = M_DOT_G_H2_KGS_PRODUZIDA # Vazão mássica de H2
# Relação Estequiométrica: 9 kg H2O / 1 kg H2 (Base Mássica)
RAZAO_H2O_CONSUMO = 9.0

# 1.1. CÁLCULO DO CONSUMO ESTEQUIOMÉTRICO
M_DOT_H2O_CONSUMIDA_KGS = M_DOT_G_H2_KGS_PRODUZIDA * RAZAO_H2O_CONSUMO # ~0.2225 kg/s

# 1.2. ÁGUA NÃO CONSUMIDA TOTAL (SEM PERDA)
M_DOT_H2O_NAO_CONSUMIDA_KGS = M_DOT_H2O_RECIRC_TOTAL_KGS - M_DOT_H2O_CONSUMIDA_KGS 

# 2. VAZÃO MÁSSICA DE O2 (Estequiometria 8:1 em massa, base O2:H2)
M_DOT_G_O2 = M_DOT_G_H2 * (31.9988 / 2.016) # ~0.19776 kg/s

# --- NOVAS CONSTANTES DE BALANÇO (Demister e Cross-over) ---
# Fator de Cross-over de Água (Massa Total de H2O no H2 = 5 x M_DOT_H2O_CONSUMIDA)
FATOR_CROSSOVER_H2 = 5.0
# Critério Demister na Saída (20 mg/Nm³ -> 0.02 g/Nm³)
LIMITE_LIQUIDO_DEMISTER_G_NM3 = 0.02
# Volume Molar Padrão
V_MOLAR_PADRAO_NM3_KMOL = 22.414 
MM_H2_CALC = 2.016 # kg/kmol
MM_O2_CALC = 31.998 # kg/kmol

# 3. DISTRIBUIÇÃO DA ÁGUA NÃO CONSUMIDA (Vazões Totais de H2O que seguem nos fluxos de Gás)
M_H2O_TOTAL_H2_KGS = M_DOT_H2O_CONSUMIDA_KGS * FATOR_CROSSOVER_H2
M_H2O_TOTAL_O2_KGS = M_DOT_H2O_NAO_CONSUMIDA_KGS - M_H2O_TOTAL_H2_KGS

# Usamos as chaves antigas com os novos valores base para compatibilidade com outros módulos
M_DOT_H2O_LIQ_IN_H2_TOTAL_KGS = M_H2O_TOTAL_H2_KGS
M_DOT_H2O_LIQ_IN_O2_TOTAL_KGS = M_H2O_TOTAL_O2_KGS
R_ARRASTE_H2 = 1.0 
R_ARRASTE_O2 = 1.0 

# --------------------------------------------------------------------------

# =================================================================
# === CONFIGURAÇÕES DE OTIMIZAÇÃO E MODOS OPERACIONAIS (FIXADOS) ===
# =================================================================
MODE_DEOXO_FINAL = 'ADIABATIC'
L_DEOXO_OTIMIZADO_M = 0.8 # m 
# ALTERADO: Fluxo sequencial Deoxo -> PSA (VSA REMOVIDO)
DC2_MODE_FINAL = 'PSA_DIRETO' 

# =================================================================
# === CONSTANTES GLOBAIS DE PROCESSO (Dimensionamento) ===
# =================================================================
T_IN_C = 60.0       # °C (Saída do Eletrolisador - Temperatura de entrada no sistema)
P_IN_BAR = 40.0     # bar (Pressão do Sistema - Pressão de entrada no sistema)

# --- TEMPERATURAS ALVO ---
T_CHILLER_OUT_H2_C_C1 = 4.0 # °C (Chiller 1 - Fluxo H2)
T_CHILLER_OUT_H2_C_C2 = -6.0 # °C (Chiller 2 - Não usado no fluxo H2 atual)
T_CHILLER_OUT_O2_C = 4.0 # °C (Chiller 1 - Fluxo O2)
T_JACKET_DEOXO_C = 50.0 # °C (Temperatura de referência da jaqueta para alertas)

# --- PRESSÕES ---
P_OUT_VALVULA_O2_BAR = 15.0 # bar (Pressão de saída da Válvula O2)
P_OUT_VALVULA_H2_BAR = 15.0 # bar (Pressão de saída da Válvula Pós-Deoxo H2 - Não usada diretamente no modelo JT)
P_DRENO_OUT_BAR = 4.0 # bar (Pressão de saída dos drenos para Flash Drum e Mixer)
# PARÂMETROS DE PRESSÃO DO VSA
P_VSA_PROD_BAR = 38.0  # Pressão de saída do produto H2
P_VSA_REG_BAR = 0.1    # Pressão de regeneração (vácuo)
P_PERDA_BAR = 0.05 # bar (Perda de pressão do Dry Cooler/Trocador de Calor Quente)

# --- VAZÃO DE REFRIGERANTE (MOVIDA DO modelo_dry_cooler.py) ---
M_DOT_REF_H2 = 5.0              # kg/s (Vazão de refrigerante para o sistema H2/DC) 
M_DOT_REF_O2 = 1.0              # kg/s (Vazão de refrigerante para o sistema O2/DC) 

# --- TAXAS DE CROSSOVER (Entrada) ---
Y_O2_IN_H2 = 0.0002 # O2 no H2: 0.02% (200 ppm)
Y_H2_IN_O2 = 0.004  # H2 no O2: 0.4% (4000 ppm)

# Dimensões do Deoxo (Referência)
L_DEOXO_ORIGINAL_M = 1.294 

# =================================================================
# === LISTAS DE COMPONENTES DO SISTEMA (ATUALIZADAS) ===
# =================================================================
# NOVO FLUXO H2: KOD 1 -> Dry Cooler 1 -> Chiller 1 -> KOD 2 -> Coalescedor 1 -> Aquecedor Imaginário -> Deoxo -> PSA
COMPONENTS_H2 = ["Entrada", "KOD 1", "Dry Cooler 1", "Chiller 1", "KOD 2", "Coalescedor 1", "Aquecedor Imaginário", "Deoxo", "PSA"]
# FLUXO O2 MANTIDO: KOD 1 -> Dry Cooler 1 -> Chiller 1 -> KOD 2 -> Coalescedor 1 -> Válvula
COMPONENTS_O2 = ["Entrada", "KOD 1", "Dry Cooler 1", "Chiller 1", "KOD 2", "Coalescedor 1", "Válvula"]

GASES = ['H2', 'O2']

# =================================================================
# === LIMITES DE ALERTA OPERACIONAL ===
# =================================================================
LIMITES = {
    'Deoxo': {
        'T_MAX_C': 60.0,            # Risco de hot spot cinético
        'y_O2_MAX': 0.025,          # Risco de explosão/hot spot
    },
    'Secador Adsorvente': { # MANTIDO para PSA/VSA usarem
        'T_MAX_C': 50.0,            # Redução da capacidade do adsorvente
        'y_H2O_MAX_PPM': 100.0,     # Umidade de entrada não deve exceder ~100 ppm
    },
    'PSA': { # Limites do PSA
         'T_MAX_C': 50.0,
         # O PSA exige pureza de entrada abaixo de 100 ppm para funcionar
         'y_H2O_MAX_PPM': 100.0,
    },
    'VSA': { # MANTIDO para compatibilidade, mas não mais usado no fluxo H2
         'T_MAX_C': 50.0,
         'y_H2O_MAX_PPM': 50000.0,
    }
}

# =================================================================
# === CONSTANTES DE LIMITE E CONVERSÃO (Para Plotagem e Cálculo) ===
# =================================================================

# Limite de Pureza Alvo (5 ppm molar para H2O)
Y_H2O_LIMIT_MOLAR = 5e-6

# Limites mássicos (Calculados na referência original, mantidos para plots)
W_H2O_LIMIT_H2_PCT = 0.0008927 # %
W_H2O_LIMIT_O2_PCT = 0.0000563 # %