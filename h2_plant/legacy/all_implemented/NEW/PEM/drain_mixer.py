# drain_mixer.py
import pandas as pd
import numpy as np
import sys
import CoolProp.CoolProp as CP 

# 💥 CORREÇÃO PRINCIPAL: Importa a função real 'mixer_model' com o alias 'modelo_mixer' 
# para corresponder ao uso em todo o arquivo e resolver o erro "cannot import name 'modelo_mixer'".
try:
    from modulos.modelo_mixer import mixer_model as modelo_mixer
    from modulos.modelo_valvula import modelo_valvula_isoentalpica
    from modulos.modelo_tank_vent import FlashDrumModel, imprimir_resultados 
except ImportError as e:
    # Se esta parte falhar, o bloco de tratamento de erro no main_simulator será executado.
    raise ImportError(f"Falha na importação de um módulo principal de dreno: {e}")

from aux_coolprop import calcular_solubilidade_gas_henry 
# Importa as novas funções de plotagem
from plots_modulos.plot_concentracao_dreno import plot_concentracao_dreno 
from plots_modulos.plot_propriedades_linha_dreno import plot_propriedades_linha_dreno 
from plots_modulos.plot_drenos_individuais import plot_drenos_individuais 
from plots_modulos.plot_recirculacao_mixer import plot_recirculacao_mixer 
# NOVO: Importa o novo plot de concentração em linha
from plots_modulos.plot_concentracao_linha_dreno import plot_concentracao_linha_dreno 

from plot_reporter_base import salvar_e_exibir_plot, exibir_estado_final_mixer, exibir_estado_recirculacao, exibir_validacao_balanco_global

# Importa as constantes exatas (agora estão em constants_and_config.py)
try:
    from constants_and_config import (
        P_IN_BAR, T_IN_C,
        M_DOT_H2O_RECIRC_TOTAL_KGS, M_DOT_H2O_CONSUMIDA_KGS, # Constantes de balanço global
        M_DOT_G_H2, M_DOT_G_O2, FATOR_CROSSOVER_H2,
        P_DRENO_OUT_BAR, 
    )
except ImportError:
    print("AVISO: Falha ao importar constantes de Dreno de Recirculação. Usando Fallback.")
    # Fallback para garantir a execução (Valores Sem Perda)
    M_DOT_H2O_RECIRC_TOTAL_KGS = 250000.0 / 3600.0 # ~69.4444 kg/s
    M_DOT_H2O_CONSUMIDA_KGS = 801.0 / 3600.0 # ~0.2225 kg/s
    M_DOT_G_H2 = 0.02472
    M_DOT_G_O2 = 0.19776
    FATOR_CROSSOVER_H2 = 5.0
    P_DRENO_OUT_BAR = 4.0


def extrair_dados_dreno(df: pd.DataFrame, gas_fluido: str):
    """
    Extrai os dados de massa, T e P de todos os drenos. 
    RESTRITO: Apenas KOD 1, KOD 2 e PEM Dreno Recirc.
    """
    
    df_entrada = df[df['Componente'] == 'Entrada'].iloc[0]
    T_pem_c = df_entrada['T_C']
    P_pem_bar = df_entrada['P_bar']
    
    if T_pem_c is None: T_pem_c = T_IN_C
    if P_pem_bar is None: P_pem_bar = P_IN_BAR
    
    drenos_list = []
    
    # 1. CÁLCULO DO DRENO DO PEM (Água que o Demister remove no PEM)
    # Valor é a água líquida em excesso retirada pelo Demister inicial (Agua_Pura_Removida_H2O_kg_s na Entrada)
    m_dot_dreno_pem = df_entrada['Agua_Pura_Removida_H2O_kg_s'] 
    
    if m_dot_dreno_pem > 0:
        drenos_list.append({
            'Componente': f'PEM Dreno Recirc.', 
            'm_dot': m_dot_dreno_pem, 
            'T': T_pem_c,
            'P_bar': P_pem_bar,
            'P_kPa': P_pem_bar * 100, 
            'gas_fluido': gas_fluido,
            'Gas_Dissolvido_in_mg_kg': 0.0, 
            'y_gas_princ': 1.0 
        })
        
    # 2. Extrair drenos dos componentes (APENAS KODS)
    
    # 🌟 RESTRIÇÃO DE REAPROVEITAMENTO: Apenas KOD 1 e KOD 2 são permitidos.
    componentes_dreno_permitidos = ['KOD 1', 'KOD 2'] 
    
    for comp in componentes_dreno_permitidos:
        if comp in df['Componente'].values:
            comp_data = df[df['Componente'] == comp].iloc[0]
            
            # A coluna 'Agua_Pura_Removida_H2O_kg_s' rastreia a massa de água pura removida pelo componente.
            m_dot_agua_pura_removida = comp_data.get('Agua_Pura_Removida_H2O_kg_s', 0.0)
            
            if m_dot_agua_pura_removida > 0:
                T_c = comp_data['T_C']
                P_bar = comp_data['P_bar']
                y_gas_princ = comp_data.get(f'y_{gas_fluido}', 1.0) 
                
                solubilidade_mg_kg = calcular_solubilidade_gas_henry(gas_fluido, T_c, P_bar, y_gas_princ)

                drenos_list.append({
                    'Componente': comp,
                    'm_dot': m_dot_agua_pura_removida,
                    'T': T_c,
                    'P_bar': P_bar,
                    'P_kPa': P_bar * 100,
                    'gas_fluido': gas_fluido,
                    'Gas_Dissolvido_in_mg_kg': solubilidade_mg_kg, 
                    'y_gas_princ': y_gas_princ
                })
                
    return drenos_list


def simular_linha_dreno(drenos_list: list, gas_fluido: str, P_alvo_bar: float):
    
    if not drenos_list:
        return None, None
        
    print(f"\n--- INICIANDO LINHA DE DRENO: {gas_fluido} (Juntar -> Válvula -> Flash Drum) ---")
    
    # 1. Pré-Mixer Virtual para Obter Média Ponderada (USANDO TODOS OS DRENOS)
    total_mass = sum(d['m_dot'] for d in drenos_list) # total_mass is in kg/s
    if total_mass == 0: 
        print(f"!!! AVISO DEBUG ({gas_fluido}): Vazão de massa total dos drenos é zero. Ignorando Flash Drum. !!!")
        return None, None
    
    # Propriedades de Entrada (média ponderada) - AGORA COM TODOS OS DRENOS
    T_in_C = sum(d['m_dot'] * d['T'] for d in drenos_list) / total_mass
    P_in_bar = min(d['P_bar'] for d in drenos_list) 
    C_in_mg_kg = sum(d['m_dot'] * d['Gas_Dissolvido_in_mg_kg'] for d in drenos_list) / total_mass
    
    T_in_K = T_in_C + 273.15
    P_in_Pa = P_in_bar * 1e5
    
    # Entrada de Gás Dissolvido
    M_gas_kg_mol = CP.PropsSI('M', gas_fluido)
    C_in_mol_L = (C_in_mg_kg / 1e6) / M_gas_kg_mol 
    
    
    # 2. Válvula (Redução de Pressão, Mantendo T Constante)
    P_out_Pa = P_alvo_bar * 1e5
    
    # --- MODIFICAÇÃO SOLICITADA: FORÇAR T_OUT = T_IN (Processo Isotérmico) ---
    T_out_C = T_in_C # T de saída é igual à T de entrada (simplificação isotérmica)
    P_out_bar = P_alvo_bar
    # -------------------------------------------------------------------------
    
    Q_L_m3_h = 0.0
    H_in_J_kg = 0.0
    
    # CÁLCULO DE VAZÃO VOLUMÉTRICA E ENTALPIA
    try:
        rho_water = CP.PropsSI('D', 'T', T_in_K, 'P', P_in_Pa, 'Water')
        Q_L_m3_h = (total_mass / rho_water) * 3600
        Q_L_m3_s = Q_L_m3_h / 3600.0 # Vazão volumétrica de líquido em m³/s
        H_in_J_kg = CP.PropsSI('H', 'T', T_in_K, 'P', P_in_Pa, 'Water')
    except:
        rho_water = 1000.0
        Q_L_m3_h = (total_mass / rho_water) * 3600
        Q_L_m3_s = Q_L_m3_h / 3600.0 # Vazão volumétrica de líquido em m³/s
        H_in_J_kg = 4186.0 * (T_in_K - 273.15) # Estimativa
    
    # --- CÁLCULO DE ENTALPIA DE SAÍDA APÓS REDUÇÃO DE PRESSÃO (A temperatura constante requer um novo H)
    # Devido à simplificação, usamos a entalpia da água na T_out e P_out (apenas para exibição)
    try:
        H_out_J_kg = CP.PropsSI('H', 'T', T_out_C + 273.15, 'P', P_out_Pa, 'Water')
    except:
        H_out_J_kg = H_in_J_kg # Fallback para Entalpia Constante, se o CoolProp falhar no novo ponto.
        
    print(f"Válvula ({P_in_bar:.2f} bar -> {P_out_bar:.2f} bar): T_out={T_out_C:.2f}°C (T CONSTANTE FORÇADA)")
    
    # ESTADO DE ENTRADA (PRÉ-PROCESSAMENTO: Pós-Agregação)
    entrada_dreno = {
        'Componente': f'Dreno {gas_fluido} - IN (Agregação)',
        'm_dot_kg_h': total_mass * 3600,
        'T': T_in_C,
        'P_bar': P_in_bar,
        'P_kPa': P_in_bar * 100,
        'h_kJ_kg': H_in_J_kg / 1000,
        'C_diss_mg_kg': C_in_mg_kg
    }
    
    # --- CORREÇÃO: VERIFICAÇÃO DE CONCENTRAÇÃO ZERO PARA EVITAR DIVISÃO POR ZERO ---
    
    if C_in_mol_L < 1e-10: 
        print(f"Aviso: Concentração de gás dissolvido ({C_in_mg_kg:.4e} mg/kg) é zero. Pulando Flash Drum. (Tank + Vent)")
        C_out_mg_kg = 0.0 
        saida_dreno = {
            'Componente': f'Dreno {gas_fluido} - OUT (Tank + Vent)',
            'm_dot_kg_h': total_mass * 3600, 
            'T': T_out_C,
            'P_bar': P_out_bar, 
            'P_kPa': P_out_bar * 100,
            'h_kJ_kg': H_out_J_kg / 1000, # Usando a entalpia calculada no novo ponto T, P
            'C_diss_mg_kg': C_out_mg_kg
        }
        exibir_tabela_processo_dreno(entrada_dreno, saida_dreno, gas_fluido)
        return entrada_dreno, saida_dreno
        
    # 3. Flash Drum (Desgaseificação) - Renomeado para Tank + Vent
    eficiencia_desejada = 0.95 
    
    # 💥 CORREÇÃO (D_tanque_m): Definir o diâmetro para passar ao construtor FlashDrumModel
    D_tanque_m_gas = 1.0 if gas_fluido == 'H2' else 1.5 
    
    # FlashDrumModel importado de modulos.modelo_tank_vent
    flash_drum = FlashDrumModel(
        T_C=T_out_C, 
        P_op_kPa=P_out_bar * 100, 
        # Passando Q_L_m3_h (m³/h). O FlashDrum.py agora lida com a conversão interna para m³/s
        Q_L_m3_h=Q_L_m3_h, 
        C_gas_in_mol_L=C_in_mol_L, 
        eficiencia_desejada=eficiencia_desejada, 
        gas_name=gas_fluido,
        D_tanque_m=D_tanque_m_gas # <--- ARGUMENTO FALTANTE ADICIONADO
    )
    
    modelagem, dimensionamento = flash_drum.simular()
    
    imprimir_resultados(f'{gas_fluido} (Após Válvula)', modelagem, dimensionamento)
    
    C_out_mol_L = modelagem.get('C_final_mol_L', 0.0) 
    M_gas_kg_mol = CP.PropsSI('M', gas_fluido)
    C_out_mg_kg = C_out_mol_L * M_gas_kg_mol * 1e6 
    
    # ESTADO DE SAÍDA (PÓS-PROCESSAMENTO, ENTRADA DO MIXER) - Renomeado para Tank + Vent
    saida_dreno = {
        'Componente': f'Dreno {gas_fluido} - OUT (Tank + Vent)',
        'm_dot_kg_h': total_mass * 3600, 
        'T': T_out_C,
        'P_bar': P_out_bar, 
        'P_kPa': P_out_bar * 100,
        'h_kJ_kg': H_out_J_kg / 1000, 
        'C_diss_mg_kg': C_out_mg_kg
    }
    
    exibir_tabela_processo_dreno(entrada_dreno, saida_dreno, gas_fluido)
    
    return entrada_dreno, saida_dreno


def exibir_tabela_drenos_raw(drenos_list: list, gas_fluido: str):
    """Exibe uma tabela detalhada dos drenos brutos, com Vazão em kg/s e kg/h."""
    
    if not drenos_list:
        print(f"\n--- DRENOS BRUTOS {gas_fluido} ---")
        print("Nenhum dreno bruto para exibir.")
        return

    df_drenos = pd.DataFrame(drenos_list).copy()
    
    # Calcula Vazão kg/h e Vazão kg/s
    df_drenos['m_dot_kg_h'] = df_drenos['m_dot'] * 3600 
    df_drenos.rename(columns={'m_dot': 'm_dot_kg_s'}, inplace=True)

    # Seleciona e renomeia colunas
    cols = ['Componente', 'm_dot_kg_s', 'm_dot_kg_h', 'T', 'P_bar', 'Gas_Dissolvido_in_mg_kg']
    df_display = df_drenos[cols].rename(columns={
        'T': 'T (°C)',
        'P_bar': 'P (bar)',
        'Gas_Dissolvido_in_mg_kg': 'Conc. Diss. (mg/kg)',
        'm_dot_kg_s': 'm_dot (kg/s)',
        'm_dot_kg_h': 'm_dot (kg/h)'
    })

    # Formatação dos dados
    df_display['m_dot (kg/s)'] = df_display['m_dot (kg/s)'].map('{:.5f}'.format) # Alta precisão para pequenos fluxos
    df_display['m_dot (kg/h)'] = df_display['m_dot (kg/h)'].map('{:.2f}'.format)
    df_display['T (°C)'] = df_display['T (°C)'].map('{:.1f}'.format)
    df_display['P (bar)'] = df_display['P (bar)'].map('{:.1f}'.format)
    df_display['Conc. Diss. (mg/kg)'] = df_display['Conc. Diss. (mg/kg)'].map('{:.4f}'.format)
    
    print("\n" + "="*100)
    print(f"TABELA DE DADOS DOS DRENOS BRUTOS (INPUTS) - FLUXO DE {gas_fluido}")
    print("="*100)
    # Requer que 'tabulate' esteja instalado
    try:
        print(df_display.to_markdown(index=False))
    except ImportError:
         print(df_display.to_string(index=False)) # Fallback
    print("="*100)
    
# --- FIM DA NOVA FUNÇÃO DE EXIBIÇÃO RAW ---


def exibir_tabela_processo_dreno(entrada: dict, saida: dict, gas_fluido: str):
    """
    Exibe uma tabela formatada da linha de dreno (Agregação IN -> Flash Drum OUT).
    """
    if not entrada or not saida:
        print(f"\n--- TABELA DE DADOS DO PROCESSO DRENO {gas_fluido} ---")
        print("Dados insuficientes para exibir a tabela.")
        print("----------------------------------------------------------------")
        return

    data = {
        'Propriedade': ['Componente', 'Vazão (kg/h)', 'Temperatura (°C)', 'Pressão (bar)', 'Entalpia (kJ/kg)', f'Conc. Diss. (mg/kg)'],
        'Agregação IN': [
            entrada['Componente'],
            entrada['m_dot_kg_h'],
            entrada['T'],
            entrada['P_bar'],
            entrada['h_kJ_kg'],
            entrada['C_diss_mg_kg']
        ],
        # Renomeação de rótulo: Tank + Vent
        'Tank + Vent OUT': [
            saida['Componente'],
            saida['m_dot_kg_h'],
            saida['T'],
            saida['P_bar'], # Chave agora é 'P_bar'
            saida['h_kJ_kg'],
            saida['C_diss_mg_kg']
        ]
    }

    df_display = pd.DataFrame(data).set_index('Propriedade')
    
    # Formatação dos floats
    df_display.iloc[1:4] = df_display.iloc[1:4].map(lambda x: f'{x:.2f}')
    df_display.iloc[4:] = df_display.iloc[4:].map(lambda x: f'{x:.4f}')

    print("\n" + "="*80)
    print(f"TABELA DE DADOS DO PROCESSO DA LINHA DE DRENO: FLUXO DE {gas_fluido}")
    print("="*80)
    try:
        print(df_display.to_markdown())
    except ImportError:
        print(df_display.to_string()) # Fallback
    print("="*80)


def simular_reposicao_agua(dreno_final_state: dict, P_out_bar: float, T_makeup_C: float = 20.0, M_dot_target_kgs: float = M_DOT_H2O_RECIRC_TOTAL_KGS):
    """
    Simula a adição de água de reposição para atingir a vazão total de recirculação (250000 kg/h).
    Usa o modelo Mixer.py para o balanço de massa e energia.
    
    Args:
        dreno_final_state (dict): Estado de saída do Mixer de Drenos.
        P_out_bar (float): Pressão de saída (4 bar).
        T_makeup_C (float): Temperatura da água de reposição (20 °C).
        M_dot_target_kgs (float): Vazão alvo de recirculação (69.444 kg/s).
    """
    m_dot_dreno_kgs = dreno_final_state.get('M_dot_H2O_final_kg_s', 0.0)
    T_dreno_C = dreno_final_state.get('T_out_C', T_IN_C)
    P_dreno_kPa = P_out_bar * 100

    m_dot_target_kgs = M_dot_target_kgs
    m_dot_makeup_kgs = m_dot_target_kgs - m_dot_dreno_kgs
    
    # Define a variável m_dot_out_kgs no escopo principal, garantindo que seja definida
    m_dot_out_kgs = m_dot_dreno_kgs 
    
    if m_dot_makeup_kgs < 1e-6:
        # Vazão de Dreno atingiu ou excedeu o alvo. Não é necessária reposição significativa.
        print("\nAVISO: Vazão de Dreno excede ou iguala a vazão alvo de Recirculação. Não é necessária reposição.")
        m_dot_makeup_kgs = 0.0
        m_dot_out_kgs = m_dot_dreno_kgs
    else:
        m_dot_out_kgs = m_dot_target_kgs
        
    T_makeup_K = T_makeup_C + 273.15
    P_makeup_kPa = P_dreno_kPa # Assumindo que a reposição é pressurizada para P_out_bar
    
    T_out_C = T_dreno_C
    H_out_J_kg = dreno_final_state.get('H_liq_out_J_kg', 0.0)
    
    if m_dot_makeup_kgs > 1e-6:
        print(f"Reposição: Adicionando {m_dot_makeup_kgs * 3600:.2f} kg/h de água a {T_makeup_C:.1f}°C.")
        
        # Misturar Dreno (Stream 1) + Reposição (Stream 2)
        input_streams_recirc = [
            {'m_dot': m_dot_dreno_kgs, 'T': T_dreno_C, 'P': P_dreno_kPa},
            {'m_dot': m_dot_makeup_kgs, 'T': T_makeup_C, 'P': P_makeup_kPa}
        ]
        
        # 💥 CORREÇÃO: Chamando a função importada como 'modelo_mixer'
        _, output_results_recirc = modelo_mixer(input_streams_recirc, P_out_kPa=P_dreno_kPa)
        
        if output_results_recirc and 'erro' not in output_results_recirc:
            T_out_C = output_results_recirc.get('Temperatura de Saída (°C) (T_3)')
            H_out_J_kg = output_results_recirc.get('Entalpia Específica de Saída (kJ/kg) (h_3)') * 1000
        else:
            # Fallback para balanço de energia simplificado (em caso de erro CoolProp)
            T_out_C = (m_dot_dreno_kgs * T_dreno_C + m_dot_makeup_kgs * T_makeup_C) / m_dot_out_kgs
            C_p_H2O = 4186.0 # J/kg.K
            H_out_J_kg = C_p_H2O * (T_out_C + 273.15 - 273.15) # Estimativa
            print(f"AVISO: Mixer de Reposição usou Balanço Simplificado. T_out={T_out_C:.2f}°C")

    # --- Organiza os dados para Exibição e Plotagem ---
    
    # Dreno Final (Antes da reposição)
    estado_dreno_final_antes = {
        'Estado': 'Dreno Final (Antes)',
        'Vazão (kg/s)': m_dot_dreno_kgs,
        'Pressão (bar)': P_out_bar,
        'Temperatura (°C)': T_dreno_C
    }
    
    # Estado Final Pós-Reposição
    estado_recirculacao_final = {
        'Estado': 'Recirculação (Depois)',
        'T_out_C': T_out_C,
        'P_out_bar': P_out_bar,
        'M_dot_out_kgs': m_dot_out_kgs,
        'H_out_J_kg': H_out_J_kg,
        'M_dot_makeup_kgs': m_dot_makeup_kgs
    }

    # DataFrame para Plotagem (df_plot_recirc)
    df_plot_recirc = pd.DataFrame({
        'Estado': [estado_dreno_final_antes['Estado'], estado_recirculacao_final['Estado']],
        'Vazão (kg/s)': [m_dot_dreno_kgs, m_dot_out_kgs],
        'Pressão (bar)': [P_out_bar, P_out_bar],
        'Temperatura (°C)': [T_dreno_C, T_out_C]
    })
    
    return estado_recirculacao_final, df_plot_recirc


def executar_simulacao_mixer(df_h2: pd.DataFrame, df_o2: pd.DataFrame, mostrar_grafico: bool = False):
    """
    Executa a lógica de extração, mistura, plotagem dos drenos e, finalmente, a reposição de água.
    """
    # Inicializa variáveis para garantir o retorno seguro, mesmo em caso de falha inicial
    drenos_h2_raw = extrair_dados_dreno(df_h2, 'H2')
    drenos_o2_raw = extrair_dados_dreno(df_o2, 'O2')
    estado_recirculacao_final = None 
    drenos_plot_data = {'H2_RAW': drenos_h2_raw, 'O2_RAW': drenos_o2_raw, 'FINAL_MIXER': {'Conc_H2_final_mg_kg': 0.0, 'Conc_O2_final_mg_kg': 0.0}}

    # 💥 CORREÇÃO: A variável modelo_mixer é agora o alias importado.
    if not callable(modelo_mixer):
        print("!!! ERRO CRÍTICO DEBUG: Mixer model não é uma função (Falha na importação). Simulação de drenos abortada. !!!")
        return drenos_plot_data, None, estado_recirculacao_final 

    print("\n" + "="*80)
    print("INICIANDO CÁLCULO DA LINHA DE DRENOS (Válvula -> Flash Drum -> Mixer)")
    print("="*80)
    
    # 🌟 Exibir Tabela RAW H2 (fluxos individuais)
    exibir_tabela_drenos_raw(drenos_h2_raw, 'H₂')
    # 🌟 Exibir Tabela RAW O2 (fluxos individuais)
    exibir_tabela_drenos_raw(drenos_o2_raw, 'O₂')

    # 2. Simula a Linha H2 (Válvula + Flash Drum)
    entrada_h2, saida_h2 = simular_linha_dreno(drenos_h2_raw, 'H2', P_DRENO_OUT_BAR)
    
    if saida_h2 is None:
         print("!!! WARNING DEBUG: Simulação da Linha H2 falhou ou vazão é zero. saida_h2 é None. !!!")
    
    # 3. Simula a Linha O2 (Válvula + Flash Drum)
    entrada_o2, saida_o2 = simular_linha_dreno(drenos_o2_raw, 'O2', P_DRENO_OUT_BAR)
    
    if saida_o2 is None:
         print("!!! WARNING DEBUG: Simulação da Linha O2 falhou ou vazão é zero. saida_o2 é None. !!!")
    
    # 4. Mixer Final (Juntar H2 e O2)
    
    input_final_mixer = []
    if saida_h2 and saida_h2.get('m_dot_kg_h', 0.0) > 0:
        # Mixer Model espera m_dot em kg/s e P em kPa
        input_final_mixer.append({'m_dot': saida_h2['m_dot_kg_h'] / 3600, 'T': saida_h2['T'], 'P': saida_h2['P_kPa']})
    if saida_o2 and saida_o2.get('m_dot_kg_h', 0.0) > 0:
        input_final_mixer.append({'m_dot': saida_o2['m_dot_kg_h'] / 3600, 'T': saida_o2['T'], 'P': saida_o2['P_kPa']})

    if not input_final_mixer:
         print("\n!!! AVISO DEBUG: Nenhuma corrente líquida VÁLIDA para o Mixer Final. Simulação de drenos concluída. !!!")
         # Retorna o dicionário de dados vazio
         return drenos_plot_data, None, estado_recirculacao_final
         
    print("\n--- [MIXER FINAL: Dreno H2 (Desgaseificado) + Dreno O2 (Desgaseificado)] ---")
    
    P_out_final_kPa = P_DRENO_OUT_BAR * 100
    # O Mixer Model calcula o estado de saída (T e h)
    # 💥 CORREÇÃO: Chamando a função importada como 'modelo_mixer'
    detailed_input_final, output_results_final = modelo_mixer(input_final_mixer, P_out_final_kPa)

    # 5. Organiza os dados para plotagem
    
    final_mixer_state = None
    drenos_plot_data = {
        'H2_RAW': drenos_h2_raw, 
        'O2_RAW': drenos_o2_raw,
        'H2_IN': entrada_h2, # Agregação IN
        'H2_OUT': saida_h2, # Tank + Vent OUT
        'O2_IN': entrada_o2, # Agregação IN
        'O2_OUT': saida_o2, # Tank + Vent OUT
        'FINAL_MIXER': None,
        'P_OUT_BAR': P_DRENO_OUT_BAR # Passando a constante de P para o plot
    }
    
    # --- Cálculo de Estado Final e Robustez contra KeyError ---

    # Vazão total
    m_dot_H2O_h2_s = saida_h2.get('m_dot_kg_h', 0.0) / 3600 if saida_h2 else 0.0
    m_dot_H2O_o2_s = saida_o2.get('m_dot_kg_h', 0.0) / 3600 if saida_o2 else 0.0
    m_dot_out_manual_s = m_dot_H2O_h2_s + m_dot_H2O_o2_s # Vazão total correta
    
    # Concentração de gás dissolvido em cada linha APÓS o Flash Drum (Tank + Vent)
    C_out_H2_dissolvido_no_H2O_da_linha_H2 = saida_h2.get('C_diss_mg_kg', 0.0) if saida_h2 else 0.0
    C_out_O2_dissolvido_no_H2O_da_linha_O2 = saida_o2.get('C_diss_mg_kg', 0.0) if saida_o2 else 0.0
    
    if m_dot_out_manual_s > 0:
        # Concentração final é a soma ponderada de massa (m_dot H2O) e concentração
        C_final_H2_mg_kg = (C_out_H2_dissolvido_no_H2O_da_linha_H2 * m_dot_H2O_h2_s) / m_dot_out_manual_s 
        C_final_O2_mg_kg = (C_out_O2_dissolvido_no_H2O_da_linha_O2 * m_dot_H2O_o2_s) / m_dot_out_manual_s 
    else:
         C_final_H2_mg_kg = 0.0
         C_final_O2_mg_kg = 0.0
         print("!!! WARNING DEBUG: Vazão total de saída do Mixer é zero. Concentrações finais zeradas. !!!")

    
    if output_results_final:
        
        # Propriedades do Mixer
        T_out = output_results_final.get('Temperatura de Saída (°C) (T_3)', np.nan)
        P_out = output_results_final.get('Pressão de Saída (kPa) (P_3)', np.nan) / 100
        h_out = output_results_final.get('Entalpia Específica de Saída (kJ/kg) (h_3)', np.nan) * 1000 # Convertido para J/kg
        
        final_mixer_state = {
            'Componente': 'Mixer Final (OUT)',
            'm_dot_kg_h': m_dot_out_manual_s * 3600, 
            'T_out_C': T_out, 
            'P_out_bar': P_out, 
            'M_dot_H2O_final_kg_s': m_dot_out_manual_s, 
            'H_liq_out_J_kg': h_out, 
            'Conc_H2_final_mg_kg': C_final_H2_mg_kg, 
            'Conc_O2_final_mg_kg': C_final_O2_mg_kg 
        }
        
        drenos_plot_data['FINAL_MIXER'] = final_mixer_state # Atualiza o dicionário de plotagem
    
    # Fallback
    elif m_dot_out_manual_s > 0:
        T_ponderada = (m_dot_H2O_h2_s * saida_h2.get('T', np.nan) + m_dot_H2O_o2_s * saida_o2.get('T', np.nan)) / m_dot_out_manual_s
        
        final_mixer_state = {
            'Componente': 'Mixer Final (OUT) - FAILED/ESTIMATED',
            'm_dot_kg_h': m_dot_out_manual_s * 3600,
            'T_out_C': T_ponderada, 
            'P_out_bar': P_DRENO_OUT_BAR,
            'M_dot_H2O_final_kg_s': m_dot_out_manual_s,
            'H_liq_out_J_kg': np.nan,
            'Conc_H2_final_mg_kg': C_final_H2_mg_kg, 
            'Conc_O2_final_mg_kg': C_final_O2_mg_kg  
        }
        drenos_plot_data['FINAL_MIXER'] = final_mixer_state

    if drenos_plot_data['FINAL_MIXER'] is None:
         drenos_plot_data['FINAL_MIXER'] = {
             'Conc_H2_final_mg_kg': 0.0,
             'Conc_O2_final_mg_kg': 0.0
         }

    # 6. Exibição do Resultado Final do Dreno (Antes da Reposição)
    
    if final_mixer_state:
        exibir_estado_final_mixer(final_mixer_state)

    
    # ----------------------------------------------------------------------
    # 7. SIMULAÇÃO DE REPOSIÇÃO DE ÁGUA E PLOTAGEM
    # ----------------------------------------------------------------------
    print("\n--- INICIANDO SIMULAÇÃO DE REPOSIÇÃO DE ÁGUA (Para recirculação) ---")
    
    if final_mixer_state and final_mixer_state.get('M_dot_H2O_final_kg_s', 0.0) > 0:
        
        T_makeup = 20.0 
        estado_recirculacao_final, df_plot_recirc = simular_reposicao_agua(
            final_mixer_state, 
            P_DRENO_OUT_BAR, 
            T_makeup_C=T_makeup, 
            M_dot_target_kgs=M_DOT_H2O_RECIRC_TOTAL_KGS
        )
        
        exibir_estado_recirculacao(estado_recirculacao_final)
        
        print("\nGerando Gráfico Comparativo da Água de Recirculação (Pós-Reposição)...")
        if plot_recirculacao_mixer is not None:
             plot_recirculacao_mixer(df_plot_recirc, mostrar_grafico)
        else:
             print("AVISO: Plotagem de recirculação desabilitada.")

    else:
        print("AVISO: Vazão de água drenada insuficiente para simular a reposição.")

    # 8. Geração de Gráficos da Linha de Drenos (Restante)
    print("\nGerando Gráficos da Linha de Drenos (3 Views)...")
    
    if drenos_h2_raw and drenos_o2_raw:
        if plot_drenos_individuais is not None:
             plot_drenos_individuais(drenos_h2_raw, drenos_o2_raw, mostrar_grafico)
        else:
             print("AVISO: Plotagem de Drenos Individuais desabilitada.")
        
    if drenos_plot_data['H2_OUT'] and drenos_plot_data['O2_OUT']:
        if 'Conc_H2_final_mg_kg' in drenos_plot_data['FINAL_MIXER']:
             # 1. Gráfico de Propriedades da Linha (Corrigido para 3 pontos, Mixer Removido)
             if plot_propriedades_linha_dreno is not None:
                plot_propriedades_linha_dreno(drenos_plot_data, mostrar_grafico)
             else:
                print("AVISO: Plotagem de Propriedades da Linha de Drenos desabilitada.")
             
             # 2. Gráfico de Concentração (Barras)
             if plot_concentracao_dreno is not None:
                plot_concentracao_dreno(drenos_plot_data, mostrar_grafico)
             else:
                print("AVISO: Plotagem de Concentração de Drenos (Barras) desabilitada.")
                
             # 3. NOVO: Gráfico de Concentração (Linha)
             if plot_concentracao_linha_dreno is not None:
                plot_concentracao_linha_dreno(drenos_plot_data, mostrar_grafico)
             else:
                print("AVISO: Plotagem de Concentração de Drenos (Linha) desabilitada.")
                
        else:
             print("AVISO: Dicionário FINAL_MIXER incompleto/inválido para plotagem de Concentração/Propriedades.")
    else:
        print("AVISO: Dados agregados insuficientes (saídas do Tank + Vent) para gerar gráficos de Linha/Concentração.")
    
    print("Gráficos de Dreno gerados com sucesso.")
    
    # 9. Validação do Balanço Global (Métrica de Controle)
    m_dot_dreno_pem_h2 = df_h2[df_h2['Componente'] == 'Entrada']['Agua_Pura_Removida_H2O_kg_s'].iloc[0]
    m_dot_dreno_pem_o2 = df_o2[df_o2['Componente'] == 'Entrada']['Agua_Pura_Removida_H2O_kg_s'].iloc[0]
    
    m_dot_agua_removida_componentes = (df_h2[df_h2['Componente'] != 'Entrada']['Agua_Pura_Removida_H2O_kg_s'].sum() + 
                                       df_o2[df_o2['Componente'] != 'Entrada']['Agua_Pura_Removida_H2O_kg_s'].sum())
    
    m_dot_drenada_total_kgs = m_dot_dreno_pem_h2 + m_dot_dreno_pem_o2 + m_dot_agua_removida_componentes
    
    exibir_validacao_balanco_global(m_dot_drenada_total_kgs, M_DOT_H2O_CONSUMIDA_KGS)
    
    return drenos_plot_data, None, estado_recirculacao_final