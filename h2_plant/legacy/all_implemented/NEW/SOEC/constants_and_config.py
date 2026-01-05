# constants_and_config.py
# Centraliza todas as constantes e configurações do sistema de tratamento de gás.

import numpy as np
import CoolProp.CoolProp as CP
import sys

# === Cálculo da Temperatura de Saturação para 5 bar ===
try:
    P_SAT_PA = 5.0 * 1e5
    T_SAT_K = CP.PropsSI('T', 'P', P_SAT_PA, 'Q', 0, 'Water')
    T_SAT_C = T_SAT_K - 273.15
except Exception:
    T_SAT_C = 151.08 # Fallback manual para 5 bar (aprox. 151.08 °C)

# =================================================================
# === CONFIGURAÇÕES GLOBAIS DO ELETROLISADOR SOEC (MODIFICADO) ===
# =================================================================
# Adaptação para 6 Módulos SOEC de 2.4 MW cada.
P_NOMINAL_KW = 6 * 2400.0 # 14400.0 kW (14.4 MW Total)
# Consumo Específico de Energia (SEC)
SEC_KWH_KG_H2 = 37.5 
# Vazão de Água de Recirculação (Loop Catódico): 6 Módulos * 704 kg/h
M_DOT_H2O_RECIRC_TOTAL_KGS = (6 * 704.0) / 3600.0 # 4224 kg/h (~1.17333 kg/s)

# =================================================================
# === CONSTANTES DE BALANÇO DE ÁGUA (BASEADO NA LÓGICA DO USUÁRIO PARA SOEC) ===
# =================================================================
# Vazão Mássica de H2 Produzida: 6 Módulos * 64 kg/h = 384 kg/h
M_DOT_G_H2_KGS_NOMINAL = (6 * 64.0) / 3600.0 # 384 kg/h (~0.10667 kg/s)
M_DOT_G_O2_KGS_NOMINAL = (6 * 512.0) / 3600.0 # 3072 kg/h (~0.85333 kg/s)

# 📌 CORREÇÃO DE IMPORTAÇÃO: Criando alias para compatibilidade com outros módulos
M_DOT_G_H2 = M_DOT_G_H2_KGS_NOMINAL
M_DOT_G_O2 = M_DOT_G_O2_KGS_NOMINAL


# Proporção Mássica H2O Consumida / H2 Produzido (3456 kg H2O / 384 kg H2) = 9.0
RAZAO_H2O_CONSUMO = 9.0 

# 1.1. CÁLCULO DO CONSUMO ESTEQUIOMÉTRICO 
M_DOT_H2O_CONSUMIDA_KGS = M_DOT_G_H2_KGS_NOMINAL * RAZAO_H2O_CONSUMO # 3456 kg/h (~0.9600 kg/s)

# ÁGUA TOTAL DE SAÍDA NO FLUXO H2 (Vapor não reagido + Arraste)
M_DOT_H2O_NAO_CONSUMIDA_KGS = M_DOT_H2O_RECIRC_TOTAL_KGS - M_DOT_H2O_CONSUMIDA_KGS # 768 kg/h (~0.21333 kg/s)

# --- NOVAS CONSTANTES DE BALANÇO E CROSSOVER ---
MM_H2_CALC = 2.016 # kg/kmol
MM_O2_CALC = 31.998 # kg/kmol

# CORREÇÃO CRÍTICA: Adicionando a Massa Molar da água (MM_H2O)
try:
    MM_H2O = CP.PropsSI('M', 'Water') # Massa Molar H2O (kg/kmol)
except:
    MM_H2O = 18.01528 # Fallback manual

# 💥 NOVOS VALORES DE CROSSOVER (PPM molar)
Y_O2_IN_H2 = 0.0002 # 200 ppm (O2 no H2)
Y_H2_IN_O2 = 0.0040 # 4000 ppm (H2 no O2)

# --- CÁLCULO DO BALANÇO DE MASSA COM CROSSOVER (Ajuste da Vazão Pura) ---

# 1. Vazão Molar Nominal (kmol/s)
F_H2_NOMINAL_KMOLS = M_DOT_G_H2_KGS_NOMINAL / MM_H2_CALC
F_O2_NOMINAL_KMOLS = M_DOT_G_O2_KGS_NOMINAL / MM_O2_CALC

# 2. Fração Molar do Gás Principal (1 - y_impureza)
Y_H2_PRINCIPAL = 1.0 - Y_O2_IN_H2 # H2 no Cátodo
Y_O2_PRINCIPAL = 1.0 - Y_H2_IN_O2 # O2 no Ânodo

# 3. Vazão Molar Total do Fluxo (kmol/s)
F_H2_TOTAL_FLUXO_KMOLS = F_H2_NOMINAL_KMOLS / Y_H2_PRINCIPAL
F_O2_TOTAL_FLUXO_KMOLS = F_O2_NOMINAL_KMOLS / Y_O2_PRINCIPAL

# 4. Vazão Molar de Gás de Crossover (kmol/s)
F_O2_CROSSOVER_KMOLS = F_H2_TOTAL_FLUXO_KMOLS * Y_O2_IN_H2
F_H2_CROSSOVER_KMOLS = F_O2_TOTAL_FLUXO_KMOLS * Y_H2_IN_O2

# 5. Vazão Mássica de Gás de Crossover (kg/s)
M_DOT_O2_CROSSOVER_KGS = F_O2_CROSSOVER_KMOLS * MM_O2_CALC # O2 que migrou para o H2
M_DOT_H2_CROSSOVER_KGS = F_H2_CROSSOVER_KMOLS * MM_H2_CALC # H2 que migrou para o O2

# 6. Vazão Mássica FINAL de Gás Principal (kg/s) - Vazão de Produção Líquida
# M_DOT_G_H2 e M_DOT_G_O2 já foram definidos acima como alias para o valor nominal.
# Se a lógica de cálculo usar o crossover (como deve ser):
M_DOT_G_H2_CALCULADO = M_DOT_G_H2_KGS_NOMINAL - M_DOT_H2_CROSSOVER_KGS # Gás H2 PURO que segue no fluxo H2
M_DOT_G_O2_CALCULADO = M_DOT_G_O2_KGS_NOMINAL - M_DOT_O2_CROSSOVER_KGS # Gás O2 PURO que segue no fluxo O2

# 📌 CORREÇÃO DE VALOR: O alias deve apontar para o valor calculado, se houver crossover.
M_DOT_G_H2 = M_DOT_G_H2_CALCULADO
M_DOT_G_O2 = M_DOT_G_O2_CALCULADO

# ----------------------------------------------------------------------------------

FATOR_CROSSOVER_H2 = M_DOT_H2O_NAO_CONSUMIDA_KGS / M_DOT_H2O_CONSUMIDA_KGS 

# REMOVIDO: LIMITE DE DEMISTER
LIMITE_LIQUIDO_DEMISTER_G_NM3 = 0.0 # G/Nm³ 
V_MOLAR_PADRAO_NM3_KMOL = 22.414 

# 3. DISTRIBUIÇÃO DA ÁGUA NÃO CONSUMIDA (Vazões Totais de H2O que seguem nos fluxos de Gás)
M_H2O_TOTAL_H2_KGS = M_DOT_H2O_NAO_CONSUMIDA_KGS 
M_H2O_TOTAL_O2_KGS = 0.0

# Usamos as chaves antigas com os novos valores base para compatibilidade com outros módulos
M_DOT_H2O_LIQ_IN_H2_TOTAL_KGS = M_H2O_TOTAL_H2_KGS 
M_DOT_H2O_LIQ_IN_O2_TOTAL_KGS = M_H2O_TOTAL_O2_KGS 
R_ARRASTE_H2 = 1.0 
R_ARRASTE_O2 = 1.0 

# 🌟 NOVO: Chute Inicial para a Vazão do Fluido Frio no Trocador de Calor (736.88 kg/h)
# 📌 VALOR ATUALIZADO: Vazão de dreno agregada para o chute inicial (0.20469 kg/s)
M_DOT_CHUTE_DRENO_TROC_KGS = 736.88 / 3600.0 # ~0.20469 kg/s
# 📌 VALOR ATUALIZADO: Temperatura de entrada é 20 °C (Água de Reposição)
T_CHUTE_DRENO_TROC_C = 20.0 # °C

# --------------------------------------------------------------------------

# =================================================================
# === CONFIGURAÇÕES DE OTIMIZAÇÃO E MODOS OPERACIONAIS (DEOXO ATIVADO) ===
# =================================================================
MODE_DEOXO_FINAL = 'NORMAL'  
# NOVO COMPRIMENTO CALCULADO
L_DEOXO_OTIMIZADO_M = 1.747    
DC2_MODE_FINAL = 'PSA' 

# =================================================================
# === CONSTANTES GLOBAIS DE PROCESSO (Dimensionamento) ===
# =================================================================
# T_OUT_SOEC: Temperatura real do gás que sai do SOEC (150C)
# 📌 ALTERADO: Aumentando a T_OUT_SOEC (e T_ALVO do Boiler) para 152.0 °C
T_OUT_SOEC = 152.0 # °C 

# P_OUT_SOEC_BAR: Pressão de saída do SOEC (ENTRADA DA PURIFICAÇÃO) - CORRIGIDO PARA 1 BAR
P_OUT_SOEC_BAR = 1.0 # bar 

# P_IN_SOEC_BAR: Pressão de entrada no SOEC (Recirculação)
P_IN_SOEC_BAR = 5.0     

# T_SAT_5BAR_C: Temperatura de Saturação da água a 5 bar (ENTRADA DO SOEC)
T_SAT_5BAR_C = T_SAT_C      

# T_IN_C e P_IN_BAR: Mantidos para compatibilidade com o CoolProp
T_IN_C = T_OUT_SOEC 
P_IN_BAR = P_OUT_SOEC_BAR 

# --- TEMPERATURAS ALVO ---
T_CHILLER_OUT_H2_C_C1 = 4.0 # °C (Chiller 1 - Fluxo H2, resfriamento profundo)
T_CHILLER_OUT_H2_C_C2 = -6.0 # °C (Chiller 2 - Não usado no fluxo H2 atual)
T_CHILLER_OUT_O2_C = 40.0 # °C (Novo alvo de intercooling/aftercooling O2)
T_CHILLER_OUT_O2_C_FINAL = 4.0 # °C 

# Dry Cooler H2 Target: 90 °C (Para forçar condensação antes do KOD 1)
T_DRY_COOLER_OUT_H2_C = 90.0 # °C 
# Dry Cooler O2 Target: 60 °C (Alvo do Aftercooler O2)
T_DRY_COOLER_OUT_O2_C = 60.0 # °C 

# Temperatura alvo do Dry Cooler 2 no fluxo H2 (40 °C)
T_DRY_COOLER_OUT_H2_C_DC2 = 40.0 # °C 

# MODIFICADO: Alvo do Chiller Estágio 1 (agora 4 °C)
T_CHILLER_OUT_H2_C_C2_NOVO = 4.0 # °C 

# MODIFICADO: Temperatura alvo do Chiller Estágios 2, 3, 4 e 5 (agora 4 °C)
T_CHILLER_OUT_H2_C_C3_NOVO = 4.0 # °C (Chiller Estágios - Alvo de 4 °C)

T_JACKET_DEOXO_C = 50.0 # °C (Temperatura de referência da jaqueta para alertas - Mantido)

# 🌟 NOVO: Temperatura Alvo de Saída da Água do Dreno no Trocador (Limite)
T_DRENO_OUT_ALVO_C = 99.0 # °C 

# --- PRESSÕES ---
P_OUT_COMPRESSOR_O2_BAR = 1.0 # bar (Pressão de saída = Pressão de entrada SOEC)
P_OUT_VALVULA_H2_BAR = 5.0 # bar (Pressão de saída da Válvula Pós-Deoxo H2 - Não usada diretamente no modelo JT)
# 📌 VALOR ATUALIZADO: Pressão de saída dos drenos/Mixer 1 agora é 1.0 bar
P_DRENO_OUT_BAR = 1.0 # bar (Pressão de saída dos drenos para Flash Drum e Mixer) 
P_COMPRESSOR_H2_OUT_BAR = 1.0 # bar (Pressão de saída = Pressão de entrada SOEC)
P_VSA_PROD_BAR = 4.0  
P_VSA_REG_BAR = 0.1    

# 🌟 NOVO: Perda de pressão do gás do processo no Dry Cooler (TQC)
P_PERDA_BAR = 0.05 # bar (50 mbar de perda de pressão para o gás)

# NOVAS CONSTANTES PARA CONTROLE DE TEMPERATURA DO COMPRESSOR
P_MAX_TEORICA_COMPRESSOR_H2_BAR = 40.0 # <--- Limite superior de busca (Ex: pressão máxima de projeto)
T_MAX_ALVO_COMPRESSOR_C = 120.0 # <--- Temperatura máxima permitida na descarga (120 °C)

# --- PRESSÃO ALVO COMPRESSOR H2 (1 ESTÁGIO) ---
P_TARGET_COMPRESSOR_H2_BAR = 2.09 # Mantido como referência de estágio, mas a lógica mudará.

# NOVAS Pressões alvo sequenciais para os 5 estágios do compressor H2 (visando 40 bar)
P_TARGET_COMPRESSOR_H2_EST1_BAR = 2.09 # bar 
P_TARGET_COMPRESSOR_H2_EST2_BAR = 5.0 # bar 
P_TARGET_COMPRESSOR_H2_EST3_BAR = 10.0 # bar
P_TARGET_COMPRESSOR_H2_EST4_BAR = 20.0 # bar
P_TARGET_COMPRESSOR_H2_EST5_BAR = 40.0 # bar (Pressão final do fluxo H2)

# NOVAS Pressões alvo para o Compressor O2
P_TARGET_COMPRESSOR_O2_EST2_BAR = 5.0 # bar
P_TARGET_COMPRESSOR_O2_EST3_BAR = 10.0 # bar
P_TARGET_COMPRESSOR_O2_EST4_BAR = 15.0 # bar

# --- TAXAS DE IMPUREZAS (Entrada) - Ajustadas para SOEC ---
Y_O2_IN_H2 = 0.0002 # 200 ppm (O2 no H2)
Y_H2_IN_O2 = 0.0040 # 4000 ppm (H2 no O2)

L_DEOXO_ORIGINAL_M = 1.294 

# =================================================================
# === LISTAS DE COMPONENTES DO SISTEMA (ADAPTADAS AO NOVO FLUXO) ===
# =================================================================
COMPONENTS_H2 = [
    "SOEC (Entrada)", "SOEC (Saída)", 
    "Trocador de Calor (Água Dreno)", # <--- NOVO COMPONENTE
    "Dry Cooler 1", "KOD 1", "Chiller 1", "KOD 2", "Coalescedor 1", 
    "Compressor H2 (Estágio 1)", 
    "Dry Cooler (Estágio 1)", 
    "Chiller (Estágio 1)", 
    # ESTÁGIO 2
    "Compressor H2 (Estágio 2)", 
    "Dry Cooler (Estágio 2)", 
    "Chiller (Estágio 2)", 
    # ESTÁGIO 3
    "Compressor H2 (Estágio 3)", 
    "Dry Cooler (Estágio 3)", 
    "Chiller (Estágio 3)",
    # ESTÁGIO 4
    "Compressor H2 (Estágio 4)",
    "Dry Cooler (Estágio 4)",
    "Chiller (Estágio 4)",
    # ESTÁGIO 5
    "Compressor H2 (Estágio 5)",
    "Dry Cooler (Estágio 5)",
    # "Chiller (Estágio (Estágio 5)", # <--- REMOVIDO: Para que o fluxo entre mais quente no Deoxo
    # ------------------------------------
    "Deoxo", "PSA"
]

COMPONENTS_O2 = [
    "SOEC (Entrada)", "SOEC (Saída)", 
    "Dry Cooler 1", 
    "Compressor O2 (Estágio 1)",
    "Dry Cooler O2 (Estágio 1)", 
    "Chiller O2", # Chiller O2 (4 °C)
    "Compressor O2 (Estágio 2)", # Compressor O2 (Estágio 2)
    "Dry Cooler O2 (Estágio 2)", # Dry Cooler O2 (Estágio 2)
    "Chiller O2 (Estágio 2)", # Chiller O2 (Estágio 2)
    "Compressor O2 (Estágio 3)", # Compressor O2 (Estágio 3)
    "Dry Cooler O2 (Estágio 3)", # Dry Cooler O2 (Estágio 3)
    "Chiller O2 (Estágio 3)", # Chiller O2 (Estágio 3)
    "Compressor O2 (Estágio 4)", # Compressor O2 (Estágio 4)
    "Dry Cooler O2 (Estágio 4)" # <--- ADICIONADO
]

GASES = ['H2', 'O2']

# =================================================================
# === LIMITES DE ALERTA OPERACIONAL (T_MAX alterados devido ao SOEC) ===
# =================================================================
LIMITES = {
    'Deoxo': {
        'T_MAX_C': 60.0,            # Risco de hot spot cinético (Deoxo opera apenas se T_IN for baixa)
        'y_O2_MAX': 0.025,          # Risco de explosão/hot spot
    },
    'Secador Adsorvente': { 
        'T_MAX_C': 50.0,            # Redução da capacidade do adsorvente (Mantido para PSA/VSA)
        'y_H2O_MAX_PPM': 100.0,     # Umidade de entrada não deve exceder ~100 ppm
    },
    'PSA': { # Limites do PSA
         'T_MAX_C': 50.0,
         'y_H2O_MAX_PPM': 100.0,
    },
    'VSA': {
         'T_MAX_C': 50.0,
         'y_H2O_MAX_PPM': 50000.0,
    },
    # NOVO LIMITE (O trocador de calor não pode ferver a água a 4 bar)
    'Trocador de Calor (Água Dreno)': {
         'T_MAX_C': T_OUT_SOEC # Limite de T_in no lado quente
    },
    # NOVO LIMITE (Apenas para simular a necessidade de resfriamento inicial)
    'Dry Cooler 1': {
        'T_MAX_C': 750.0 # Para resfriamento inicial (Hot Gas)
    },
    # NOVOS LIMITES (Dry Cooler H2)
    'Dry Cooler (Estágio 1)': {
        'T_MAX_C': 750.0 
    },
    'Chiller (Estágio 1)': {
        'T_MAX_C': 750.0
    },
    # NOVO LIMITE ESTÁGIO 2
    'Dry Cooler (Estágio 2)': { 
        'T_MAX_C': 750.0 
    },
    'Chiller (Estágio 2)': { 
        'T_MAX_C': 750.0
    },
    # NOVO LIMITE ESTÁGIO 3
    'Dry Cooler (Estágio 3)': { 
        'T_MAX_C': 750.0 
    },
    'Chiller (Estágio 3)': { 
        'T_MAX_C': 750.0
    },
    # NOVO LIMITE ESTÁGIO 4
    'Compressor H2 (Estágio 4)': {
         'T_MAX_C': 120.0
    },
    'Dry Cooler (Estágio 4)': {
         'T_MAX_C': 750.0
    },
    'Chiller (Estágio 4)': { 
         'T_MAX_C': 750.0
    },
    # NOVO LIMITE ESTÁGIO 5
    'Compressor H2 (Estágio 5)': { 
         'T_MAX_C': 120.0
    },
    'Dry Cooler (Estágio 5)': { 
         'T_MAX_C': 750.0
    },
    'Chiller (Estágio 5)': { # <--- LIMITE REMOVIDO DA LISTA DE COMPONENTES
         'T_MAX_C': 750.0
    },
    # NOVO LIMITE: Compressores H2 (Todos os estágios)
    'Compressor H2 (Estágio 1)': {
         'T_MAX_C': 120.0
    },
    'Compressor H2 (Estágio 2)': {
         'T_MAX_C': 120.0
    },
    'Compressor H2 (Estágio 3)': {
         'T_MAX_C': 120.0
    },
    'Compressor H2 (Estágio 4)': {
         'T_MAX_C': 120.0
    },
    'Compressor H2 (Estágio 5)': {
         'T_MAX_C': 120.0
    },
    # NOVO LIMITE O2
    'Compressor O2 (Estágio 1)': { 
         'T_MAX_C': 120.0 
    },
    # NOVO LIMITE O2 Dry Cooler
    'Dry Cooler O2 (Estágio 1)': { 
         'T_MAX_C': 60.0 
    },
    # NOVO LIMITE O2 Chiller
    'Chiller O2': {
         'T_MAX_C': 750.0 
    },
    # NOVO LIMITE O2 Compressor Estágio 2
    'Compressor O2 (Estágio 2)': { 
         'T_MAX_C': 120.0 
    },
    # NOVO LIMITE O2 Dry Cooler Estágio 2
    'Dry Cooler O2 (Estágio 2)': { 
         'T_MAX_C': 60.0 
    },
    # NOVO LIMITE O2 Chiller Estágio 2
    'Chiller O2 (Estágio 2)': { 
         'T_MAX_C': 750.0 
    },
    # NOVO LIMITE O2 Compressor Estágio 3
    'Compressor O2 (Estágio 3)': { 
         'T_MAX_C': 120.0 
    },
    # NOVO LIMITE O2 Dry Cooler Estágio 3
    'Dry Cooler O2 (Estágio 3)': { 
         'T_MAX_C': 60.0 
    },
    # NOVO LIMITE O2 Chiller Estágio 3
    'Chiller O2 (Estágio 3)': { 
         'T_MAX_C': 750.0 
    },
    # NOVO LIMITE O2 Compressor Estágio 4
    'Compressor O2 (Estágio 4)': { 
         'T_MAX_C': 120.0 
    },
    # NOVO LIMITE O2 Dry Cooler Estágio 4
    'Dry Cooler O2 (Estágio 4)': { 
         'T_MAX_C': 60.0 
    }
}

# =================================================================
# === CONSTANTES DE LIMITE E CONVERSÃO (Para Plotagem e Cálculo) ===
# =================================================================

Y_H2O_LIMIT_MOLAR = 5e-6
W_H2O_LIMIT_H2_PCT = 0.0008927 # %
W_H2O_LIMIT_O2_PCT = 0.0000563 # %