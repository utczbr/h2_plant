import numpy as np
import pickle 
from numpy.polynomial import polynomial as P 
import warnings
import sys
import os

# Adiciona o diretório raiz ao path para garantir importação correta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.interpolate import interp1d
from h2_plant.config.constants_physics import PEMConstants
from h2_plant.models import pem_physics as phys

# Ignorar warnings de otimização
warnings.filterwarnings("ignore") 

CONST = PEMConstants()

# ==========================================================
# PREPARAÇÃO DA TABELA DE DEGRADAÇÃO (CORREÇÃO DELTA)
# ==========================================================

# 1. Carregar tabelas
T_OP_H_TABLE = np.array(CONST.DEGRADATION_TABLE_YEARS) * 8760.0
raw_v_table = np.array(CONST.DEGRADATION_TABLE_V_CELL)

# 2. Normalização de Segurança: 
# Se os valores forem > 10V (provavelmente tensão de Stack), divide pelo número de células.
# Se forem < 10V, assume que já são tensão de célula.
if raw_v_table[0] > 10.0:
    V_CELL_TABLE = raw_v_table / CONST.N_cell_per_stack
else:
    V_CELL_TABLE = raw_v_table

# 3. Lógica do DELTA (CRÍTICO):
# Calculamos apenas o quanto a tensão SUBIU em relação ao início da tabela.
# Isso isola a física (Nernst/Ohm) da degradação empírica.
V_CELL_BASE_TABLE = V_CELL_TABLE[0]
V_CELL_DELTA = V_CELL_TABLE - V_CELL_BASE_TABLE

# 4. Interpolador do Delta
v_cell_deg_interpolator = interp1d(
    T_OP_H_TABLE, 
    V_CELL_DELTA, 
    kind='linear', 
    fill_value="extrapolate" # Permite prever além de 10 anos se necessário
)

def calculate_U_deg_from_table(t_op_h):
    """Retorna a penalidade de tensão (V) a ser somada à física."""
    return np.maximum(0.0, v_cell_deg_interpolator(t_op_h))

def calculate_Vcell(j, T, P_op, t_op_h):
    # Base física (Nernst + Ativação + Ohm + Conc)
    V_base = phys.calculate_Vcell_base(j, T, P_op)
    # Penalidade por tempo (Degradação)
    U_deg = calculate_U_deg_from_table(t_op_h)
    return V_base + U_deg

def P_input_system(j, T, P_op, t_op_h):
    I_total = j * CONST.Area_Total
    V_cell = calculate_Vcell(j, T, P_op, t_op_h)
    P_stack = I_total * V_cell
    # BoP Model
    P_BoP = CONST.P_bop_fixo + CONST.k_bop_var * P_stack
    return P_stack + P_BoP

# ==========================================================
# EXECUÇÃO DO PRÉ-CÁLCULO (PIECEWISE)
# ==========================================================

if __name__ == "__main__":
    print(f"Iniciando Pré-cálculo Piecewise de {CONST.H_SIM_YEARS} Anos...")

    num_meses = int(CONST.H_SIM_TOTAL_PRECALC / CONST.H_MES)
    piecewise_models_list = []  

    # Aumentado para melhor resolução em baixa carga
    j_char_points = 600 
    j_char_range = np.linspace(0.001 * CONST.j_nom, 1.3 * CONST.j_nom, j_char_points)

    # Definição do Ponto de Corte (Split)
    P_SPLIT_VALUE = 0.20 * CONST.P_nominal_sistema_W 

    for mes in range(num_meses):
        t_op_h_mes = mes * CONST.H_MES
        
        # 1. Gerar curva real para este mês
        P_input_char_W = np.array([P_input_system(j, CONST.T_default, CONST.P_op_default, t_op_h_mes) for j in j_char_range])
        
        # 2. Filtrar dados válidos (Apenas acima do consumo fixo do BoP)
        valid_indices = P_input_char_W > (CONST.P_bop_fixo * 1.001)
        P_valid = P_input_char_W[valid_indices]
        j_valid = j_char_range[valid_indices]

        if len(P_valid) < 10:
            print(f"⚠️ Aviso: Mês {mes} tem poucos pontos válidos. Verifique os limites.")
            continue

        # 3. Separar as regiões (Low vs High)
        mask_low = P_valid <= P_SPLIT_VALUE
        mask_high = P_valid > P_SPLIT_VALUE

        # 4. Ajustar Polinômios
        if np.sum(mask_low) > 5:
            coeffs_low = np.polyfit(P_valid[mask_low], j_valid[mask_low], 5)
            poly_low = np.poly1d(coeffs_low)
        else:
            poly_low = np.poly1d([0]) 

        if np.sum(mask_high) > 4:
            coeffs_high = np.polyfit(P_valid[mask_high], j_valid[mask_high], 4)
            poly_high = np.poly1d(coeffs_high)
        else:
            poly_high = np.poly1d([0])

        # 5. Armazenar
        model_dict = {
            'mes': mes,
            't_op_h': t_op_h_mes,
            'split_point': P_SPLIT_VALUE,
            'poly_low': poly_low,
            'poly_high': poly_high
        }
        
        piecewise_models_list.append(model_dict)
        
        if mes % 12 == 0:
            U_deg = calculate_U_deg_from_table(t_op_h_mes)
            print(f"  > Mês {mes}: Degradação +{U_deg*1000:.1f}mV | Split @ {P_SPLIT_VALUE/1e6:.2f}MW")

    # SALVAR ARQUIVO
    file_name = "degradation.pkl" # Nome padronizado com o Componente
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(piecewise_models_list, f)
        print(f"\n✅ Pré-cálculo PIECEWISE concluído! Arquivo '{file_name}' salvo com sucesso.")
    except Exception as e:
        print(f"\n❌ Erro ao salvar: {e}")