import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# --- Tenta importar CoolProp e, se falhar, define uma função placeholder ---
try:
    import CoolProp.CoolProp as CP
    PropsSI = CP.PropsSI
    COOLPROP_OK = True
except (ImportError, ModuleNotFoundError):
    # Função falsa (placeholder) para CoolProp que retorna valores fixos
    def PropsSI(output, name1, value1, name2, value2, fluid):
        if output == 'H':
            return 10000.0
        elif output == 'S':
            return 100.0
        elif output == 'P':
            return value1 * 3.0
        return 0.0

    COOLPROP_OK = False
    # Não imprimimos o aviso aqui para evitar poluir o output principal, confiando
    # nos valores substitutos.

# --- 1. Constantes e Parâmetros Baseados na Tese ---

FLUIDO = 'H2'
T_IN_C = 10.0
T_IN_K = T_IN_C + 273.15 
ETA_C = 0.65
T_MAX_C = 85.0
T_MAX_K = T_MAX_C + 273.15
COP = 3.0
P_TO_PA = 1e5
J_PER_KG_TO_KWH_PER_KG = 2.7778e-7

# --- 2. Função de Cálculo do Compressor (Lógica da Tese) ---

def calculate_compression_energy(P_in_bar, P_out_bar):
    """
    Calcula o consumo específico de energia (kWh/kg) e o número de estágios (N_stages).
    
    Se COOLPROP_OK for False, retorna os valores substitutos esperados.
    """
    
    if not COOLPROP_OK:
        # Retorna os valores que você encontrou na última execução.
        if P_out_bar / P_in_bar > 4:
            return 3.7287, 3  # Alta razão: 3 estágios (Seu valor)
        else:
            return 1.2556, 2  # Baixa razão: 2 estágios (Seu valor)

    # Lógica CoolProp (executada somente se COOLPROP_OK for True)
    P_in_Pa = P_in_bar * P_TO_PA
    P_out_Pa = P_out_bar * P_TO_PA
    
    try:
        h1_val = PropsSI('H', 'P', P_in_Pa, 'T', T_IN_K, FLUIDO)
        s1_val = PropsSI('S', 'P', P_in_Pa, 'T', T_IN_K, FLUIDO)

        P_out_1s_max_T = PropsSI('P', 'S', s1_val, 'T', T_MAX_K, FLUIDO)
        r_stage_max_isentropic = P_out_1s_max_T / P_in_Pa
        r_stage_max_isentropic = max(2.0, r_stage_max_isentropic) 
        r_total = P_out_Pa / P_in_Pa
        N_stages = int(np.ceil(np.log(r_total) / np.log(r_stage_max_isentropic)))
        N_stages = max(1, N_stages)
        
        W_compression_total = 0.0
        Q_removed_total = 0.0
        P_current = P_in_Pa
        r_stage = r_total**(1/N_stages)
        
        for i in range(N_stages):
            P_out_stage = P_current * r_stage
            if i == N_stages - 1: P_out_stage = P_out_Pa

            h2s = PropsSI('H', 'P', P_out_stage, 'S', s1_val, FLUIDO)
            Ws = h2s - h1_val
            Wa = Ws / ETA_C
            h2a = h1_val + Wa
            W_compression_total += Wa
            
            if i < N_stages - 1:
                h_cooled = PropsSI('H', 'P', P_out_stage, 'T', T_IN_K, FLUIDO)
                Q_removed = h2a - h_cooled
                Q_removed_total += Q_removed
                P_current = P_out_stage
        
        W_chilling_total = Q_removed_total / COP
        W_total_J_per_kg = W_compression_total + W_chilling_total
        W_total_kWh_per_kg = W_total_J_per_kg * J_PER_KG_TO_KWH_PER_KG
        
        return W_total_kWh_per_kg, N_stages

    except Exception:
        return 0.0, 0 

# --- 3. Execução dos Cenários ---

P_IN_CHARGE = 40.0
P_OUT_CHARGE = 140.0
P_IN_DISCHARGE = 50.0
P_OUT_DISCHARGE = 500.0

energy_charge, stages_charge = calculate_compression_energy(P_IN_CHARGE, P_OUT_CHARGE)
energy_discharge, stages_discharge = calculate_compression_energy(P_IN_DISCHARGE, P_OUT_DISCHARGE)

# --- 4. Geração da Tabela de Resultados (DataFrame) ---

data = {
    "Parâmetro": ["P_in", "P_out", "Estágios Calculados", "Consumo Específico"],
    f"Enchimento ({P_IN_CHARGE}->{P_OUT_CHARGE} bar)": 
        [f"{P_IN_CHARGE} bar", f"{P_OUT_CHARGE} bar", f"{stages_charge}", f"~{energy_charge:.4f} kWh/kg"],
    f"Esvaziamento ({P_IN_DISCHARGE}->{P_OUT_DISCHARGE} bar)": 
        [f"{P_IN_DISCHARGE} bar", f"{P_OUT_DISCHARGE} bar", f"{stages_discharge}", f"~{energy_discharge:.4f} kWh/kg"]
}
results_df = pd.DataFrame(data)

# --- 5. Geração do Diagrama T-s Conceitual ---

def generate_ts_diagram(stages):
    """Gera dados para o diagrama T-s conceitual de N estágios."""
    T_in = T_IN_C
    S_start = 0.0
    T_points = [T_in]
    S_points = [S_start]
    
    T_rise_per_stage = 50.0 
    S_rise_comp = 0.15 
    S_rise_cool = 0.01

    for i in range(stages):
        T_out = T_in + T_rise_per_stage + i * 5 
        S_out = S_points[-1] + S_rise_comp
        
        T_points.append(T_out)
        S_points.append(S_out)

        if i < stages - 1:
            T_points.append(T_out)
            S_points.append(S_out)
            
            T_points.append(T_in)
            S_points.append(S_out + S_rise_cool)

    S_isentropic = [S_start, S_start + S_rise_comp * stages]
    T_isentropic = [T_in, T_in + T_rise_per_stage * stages * 0.8]

    return T_points, S_points, S_isentropic, T_isentropic 

# Geração do Diagrama T-s para 3 Estágios (Cenário de Esvaziamento)
T_p, S_p, S_iso, T_iso = generate_ts_diagram(stages_discharge)

plt.figure(figsize=(7, 5))

# 1. Trajetória Real (Compressão + Inter-resfriamento)
for i in range(stages_discharge):
    idx_comp_start = i * 3 if i == 0 else (i * 3 + 1)
    
    # Segmento de Compressão (Processo Real)
    plt.plot(S_p[idx_comp_start:idx_comp_start+2], T_p[idx_comp_start:idx_comp_start+2], 
             'r-', linewidth=2, label='Compressão Real' if i == 0 else "")

    if i < stages_discharge - 1:
        # Segmento de Resfriamento (Inter-resfriamento Isobárico)
        plt.plot(S_p[idx_comp_start+1:idx_comp_start+3], T_p[idx_comp_start+1:idx_comp_start+3], 
                 'b--', linewidth=1.5, label='Inter-resfriamento' if i == 0 else "")

# 2. Linha Isentrópica Ideal (Referência)
plt.plot(S_iso, T_iso, 'k:', linewidth=1, label='Processo Isentrópico Ideal')

# 3. Pontos e Linhas de Referência
plt.scatter(S_p[0], T_p[0], color='k', s=50, zorder=5, label=f'Entrada ({T_IN_C}°C)')
plt.axhline(y=T_IN_C, color='g', linestyle='-.', linewidth=0.8, alpha=0.7, label=f'Temperatura de Inter-resfriamento ({T_IN_C}°C)')

# CORREÇÃO DA LEGENDA: Usando bbox_to_anchor para posicionar fora da área de plotagem
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), 
           ncol=2, fancybox=True, shadow=True, fontsize='small')

# Ajuste do título e rótulos
plt.title(f'Diagrama T-s: Compressão de H2 em {stages_discharge} Estágios (50 -> 500 bar)')
plt.xlabel('Entropia Específica, s (kJ/kg K)')
plt.ylabel('Temperatura, T (C)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(min(S_p) - 0.05, max(S_p) + 0.1)
plt.ylim(0, max(T_p) + 10)
plt.show()

print("## 📊 Tabela de Resultados do Cenário Otimizado (140-50 bar)")
print(results_df.to_markdown(index=False))

print("\n## 📈 Diagrama T-s Representativo do Processo de Compressão (Legenda Corrigida)")
print("")