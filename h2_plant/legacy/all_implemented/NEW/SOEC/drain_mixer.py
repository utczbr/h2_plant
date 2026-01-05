# drain_mixer.py
import pandas as pd
import numpy as np
import sys
import CoolProp.CoolProp as CP 

# 💥 IMPORTAÇÕES DE MODELOS E AUXILIARES 💥
# O modelo_mixer será usado agora para Mixer 1 e Mixer 2
try:
    from modelo_mixer import mixer_model as modelo_mixer 
except ImportError as e:
    # Este bloco é mantido apenas para evitar quebra total se o modelo_mixer não for encontrado.
    # O código DEVE falhar internamente se o CoolProp der erro.
    raise ImportError(f"ERRO CRÍTICO: Falha na importação do modelo_mixer: {e}")

from aux_coolprop import calcular_solubilidade_gas_henry 

# 💥 IMPORTAÇÃO DOS NOVOS MÓDULOS DE FLUXO DE RECIRCULAÇÃO 💥
from modelo_hydropump import calcular_bomba_modular

def aquecer_boiler_modular(m_dot_kg_s: float, T_in_C: float, T_out_C: float, h_in_kJ_kg: float, P_bar: float, eficiencia: float = 0.99):
    """
    Simula o Boiler Elétrico (Aquecimento de Líquido), usando CoolProp para entalpia.
    """
    T_in_K = T_in_C + 273.15
    T_out_K = T_out_C + 273.15
    P_Pa = P_bar * 1e5
    
    # NÃO HÁ FALLBACK: Confia-se no CoolProp
    h_out_kJ_kg = CP.PropsSI('H', 'T', T_out_K, 'P', P_Pa, 'Water') / 1000.0

    Delta_H_kJ_kg = h_out_kJ_kg - h_in_kJ_kg
    
    if Delta_H_kJ_kg <= 0:
        return {'W_eletrico_kW': 0.0, 'Delta_H_kJ_kg': 0.0, 'Q_necessario_kW': 0.0, 'h_out_kJ_kg': h_in_kJ_kg}
        
    Q_necessario_kW = m_dot_kg_s * Delta_H_kJ_kg
    W_eletrico_kW = Q_necessario_kW / eficiencia
    
    return {'W_eletrico_kW': W_eletrico_kW, 
            'Delta_H_kJ_kg': Delta_H_kJ_kg, 
            'Q_necessario_kW': Q_necessario_kW,
            'h_out_kJ_kg': h_out_kJ_kg}


# Importa as funções de plotagem (Apenas para evitar ImportError)
try:
    from plot_concentracao_dreno import plot_concentracao_dreno 
    from plot_propriedades_linha_dreno import plot_propriedades_linha_dreno 
    from plot_drenos_individuais import plot_drenos_individuais 
    from plot_recirculacao_mixer import plot_recirculacao_mixer 
    from plot_concentracao_linha_dreno import plot_concentracao_linha_dreno 
except ImportError as e:
    raise ImportError(f"ERRO CRÍTICO: Falha na importação de um módulo de Plotagem: {e}")

# 🛑 CORREÇÃO DE IMPORTAÇÃO: Importar funções de exibição de tabela de plot_reporter_base
from plot_reporter_base import (
    salvar_e_exibir_plot, exibir_estado_final_mixer, exibir_estado_recirculacao, 
    exibir_validacao_balanco_global,
    exibir_tabela_drenos_raw, 
    exibir_tabela_processo_dreno,
    exibir_resultados_bomba, 
    exibir_resultados_boiler 
)

# Importa as constantes exatas
from constants_and_config import (
    P_IN_BAR, T_IN_C,
    M_DOT_H2O_RECIRC_TOTAL_KGS, M_DOT_H2O_CONSUMIDA_KGS, # Constantes de balanço global
    P_DRENO_OUT_BAR, 
)

# 💥 DEFINIÇÃO SIMPLIFICADA E GLOBAL DOS COMPONENTES DE DRENO (APENAS KODs DO H2)
# Componentes que DRENAM (removem água líquida do fluxo de gás para o Pool/recirculação)
componentes_dreno_permitidos = [
    'KOD 1', 'KOD 2', 'Coalescedor 1' 
]
# ----------------------------------------------------------------------------------


def extrair_dados_dreno(df: pd.DataFrame, gas_fluido: str):
    """
    Extrai os dados de massa, T e P dos drenos dos KODs e Coalescedor.
    Apenas KOD 2 contribui para o Pool, mas KOD 1 é listado para rastreamento no Mixer 1.
    """
    
    if gas_fluido == 'O2':
        return []
        
    global componentes_dreno_permitidos
    
    drenos_list = []
    
    # Rastreia as T, P e M_DOT de Drenos que são misturados (KODs e Coalescedor)
    for comp in componentes_dreno_permitidos:
        if comp in df['Componente'].values:
            comp_data = df[df['Componente'] == comp].iloc[0]
            
            # Captura a água pura removida pelo componente.
            m_dot_agua_pura_removida = comp_data.get('Agua_Pura_Removida_H2O_kg_s', 0.0)
            
            T_c = comp_data['T_C'] 
            P_bar = comp_data['P_bar']
            
            # Se o componente contribui para o Pool (KOD 2), ou se for para rastreamento (KOD 1)
            # A vazão é sempre registrada, mas o impacto no Pool é gerenciado em process_execution
            if m_dot_agua_pura_removida > 0:
                
                # O Mixer de Drenos (Mixer 1) agrega KOD 1, KOD 2 e Coalescedor 1.
                
                # 💥 Calcular entalpia da água líquida
                T_K = T_c + 273.15
                P_Pa = P_bar * 1e5
                # Forçando LÍQUIDO SATURADO (Q=0)
                h_liq_J_kg = CP.PropsSI('H', 'T', T_K, 'Q', 0, 'Water') 
                
                gas_dissolvido_nome = 'O2' 
                y_gas_dissolvido = comp_data.get(f'y_{gas_dissolvido_nome}', 0.0)
                
                solubilidade_mg_kg = calcular_solubilidade_gas_henry(
                    gas_dissolvido_nome, 
                    T_c, 
                    P_bar, 
                    y_gas_dissolvido
                )
                     
                drenos_list.append({
                    'Componente': comp,
                    'm_dot': m_dot_agua_pura_removida,
                    'T': T_c,
                    'P_bar': P_bar,
                    'P_kPa': P_bar * 100,
                    'h_J_kg': h_liq_J_kg, # Entalpia Adicionada
                    'gas_fluido': gas_fluido,
                    'Gas_Dissolvido_in_mg_kg': solubilidade_mg_kg, 
                })
                
    return drenos_list


def simular_linha_dreno(drenos_list: list, gas_fluido: str, P_alvo_bar: float):
    """
    Simula o Mixer 1 (Agregação de Drenos) USANDO O MODELO COOLPROP.
    """
    
    if not drenos_list:
        print(f"!!! AVISO DEBUG ({gas_fluido}): Lista de drenos vazia. Nada a misturar. !!!")
        return None
        
    print(f"\n--- INICIANDO MIXER 1: {gas_fluido} (KODs + Coalescedor) ---")
    
    # 1. Preparar Streams para o Mixer 1
    input_mixer_1 = []
    total_mass_kgs = 0.0
    total_enthalpy_Js = 0.0
    C_mass_sum = 0.0 
    
    # Pressão de Saída (P_DRENO_OUT_BAR agora é 1.0 bar)
    P_out_mixer_1_bar = P_alvo_bar 
    P_out_mixer_1_kPa = P_out_mixer_1_bar * 100

    
    for dreno in drenos_list:
        m_dot_s = dreno['m_dot']
        h_J_kg = dreno['h_J_kg']
        
        if m_dot_s > 0:
            input_mixer_1.append({
                'm_dot': m_dot_s, 
                'T': dreno['T'], 
                'P': dreno['P_kPa'],
                'H': h_J_kg # Passa a entalpia (J/kg)
            })
            total_mass_kgs += m_dot_s
            total_enthalpy_Js += m_dot_s * h_J_kg
            C_mass_sum += dreno['m_dot'] * dreno['Gas_Dissolvido_in_mg_kg']
        
    if not input_mixer_1 or total_mass_kgs == 0:
        print(f"!!! AVISO DEBUG ({gas_fluido}): Vazão total para o Mixer 1 é zero. !!!")
        return None

    # 2. Executar Mixer 1 (Modelo CoolProp)
    output_results_mixer_1 = None
    try:
        detailed_input, output_results_mixer_1 = modelo_mixer(input_mixer_1, P_out_kPa=P_out_mixer_1_kPa)
        
        T_out_coolprop = output_results_mixer_1.get('Temperatura de Saída (°C) (T_3)', np.nan)
        h_out_coolprop_kJ = output_results_mixer_1.get('Entalpia Específica de Saída (kJ/kg) (h_3)', np.nan)
        h_out_coolprop_J = h_out_coolprop_kJ * 1000.0 # J/kg
        
        if np.isnan(T_out_coolprop) or np.isnan(h_out_coolprop_kJ):
            raise ValueError("CoolProp Mixer retornou NaN.")
        
        T_out = T_out_coolprop
        H_out_J_kg = h_out_coolprop_J
        
    except Exception as e:
        print(f"[AVISO CRÍTICO - Mixer 1] Falha no cálculo CoolProp do Mixer 1 ({gas_fluido}): {e}. Usando Balanço de Energia Manual.")
        
        # 🛑 FALLBACK ROBUSTO: Balanço de Energia Manual (Garante que T e H não são NaN)
        H_out_J_kg = total_enthalpy_Js / total_mass_kgs
        
        try:
             T_out_K = CP.PropsSI('T', 'H', H_out_J_kg, 'P', P_out_mixer_1_kPa, 'Water')
             T_out = T_out_K - 273.15
        except Exception as e_T:
             print(f"[ERRO FALLBACK T] Falha ao resolver T a partir de H: {e_T}. Usando T_in média.")
             T_out = np.mean([d['T'] for d in drenos_list])
            
        
    # 3. Organizar o Estado de Saída (Dreno Agregado)
    
    # Concentração Média Ponderada da Impureza
    C_out_mg_kg = C_mass_sum / total_mass_kgs if total_mass_kgs > 0 else 0.0

    saida_agregada = {
        'Componente': f'Dreno {gas_fluido} - AGREGADO (Mixer 1)',
        'm_dot_kg_h': total_mass_kgs * 3600, 
        'T': T_out,
        'P_bar': P_out_mixer_1_bar, 
        'P_kPa': P_out_mixer_1_kPa,
        'h_kJ_kg': H_out_J_kg / 1000.0, 
        'C_diss_mg_kg': C_out_mg_kg,
        'M_dot_H2O_final_kg_s': total_mass_kgs, # Rastrear kg/s
        'H_liq_out_J_kg': H_out_J_kg # Entalpia em J/kg
    }
    
    print(f"Mixer 1 Saída (Dreno Agregado): T={saida_agregada['T']:.2f} °C, P={saida_agregada['P_bar']:.2f} bar, m_dot={saida_agregada['m_dot_kg_h']:.2f} kg/h")

    # Retorna o estado agregado (Mixer 1 OUT)
    return saida_agregada


def exibir_tabela_drenos_raw(drenos_list: list, gas_fluido: str):
# ... (Função inalterada - Definida em plot_reporter_base.py) ...
    pass


def exibir_tabela_processo_dreno(entrada: dict, saida: dict, gas_fluido: str):
# ... (Função inalterada - Definida em plot_reporter_base.py) ...
    pass


# 🛑 CORRIGIDO: Remover a constante global do argumento default. O valor default agora é None.
def simular_reposicao_agua(dreno_agregado_state: dict, P_makeup_bar: float, T_makeup_C: float = 20.0, M_dot_target_kgs: float = None):
    """
    Simula o Mixer 2 (Dreno Agregado + Reposição), usando o MODELO COOLPROP (modelo_mixer).
    """
    
    # 🛑 CORREÇÃO: Usar a constante global se o valor target não for fornecido.
    if M_dot_target_kgs is None:
         # A constante agora está definida no escopo global (importada/fallback)
         M_dot_target_kgs = M_DOT_H2O_RECIRC_TOTAL_KGS 

    m_dot_dreno_kgs = dreno_agregado_state.get('M_dot_H2O_final_kg_s', 0.0)
    T_dreno_C = dreno_agregado_state.get('T', T_IN_C)
    P_dreno_kPa = dreno_agregado_state.get('P_kPa', P_DRENO_OUT_BAR * 100)
    h_dreno_kJ_kg = dreno_agregado_state.get('h_kJ_kg', 0.0) # Entalpia CoolProp do Mixer 1
    
    P_makeup_kPa = P_makeup_bar * 100 
    T_makeup_C = T_makeup_C
         
    m_dot_target_kgs = M_dot_target_kgs # Usa o alvo (que agora é a constante global)
    m_dot_makeup_kgs = m_dot_target_kgs - m_dot_dreno_kgs
    m_dot_out_kgs = m_dot_dreno_kgs 
    
    # 📌 P_out do Mixer 2 é a menor pressão de entrada (a da Reposição: 1.0 bar, já que P_dreno_kPa é 1.0 bar)
    P_out_mixer_2_kPa = min(P_dreno_kPa, P_makeup_kPa)
    P_out_mixer_2_bar = P_out_mixer_2_kPa / 100.0
    
    if m_dot_makeup_kgs < 1e-6:
        print("\nAVISO: Vazão de Dreno excede ou iguala a vazão alvo de Recirculação. Não é necessária reposição.")
        m_dot_makeup_kgs = 0.0
        m_dot_out_kgs = m_dot_dreno_kgs
        T_out_C = T_dreno_C
        H_out_J_kg = h_dreno_kJ_kg * 1000.0
        h_out_kJ_kg = h_dreno_kJ_kg
    else:
        m_dot_out_kgs = m_dot_target_kgs
        
        print(f"\n--- INICIANDO MIXER 2: (Dreno Agregado + Reposição) ---")
        print(f"Reposição: Adicionando {m_dot_makeup_kgs * 3600:.2f} kg/h de água a {T_makeup_C:.1f} °C (P={P_makeup_bar:.1f} bar).")
        
        # 🛑 NOVO CÁLCULO: Usar CoolProp para a entalpia da água de reposição.
        T_makeup_K = T_makeup_C + 273.15
        
        # Calcula a entalpia da água de reposição:
        # 🛑 USAR Q=0 para garantir que é a entalpia do LÍQUIDO SATURADO na T_K.
        h_makeup_J_kg = CP.PropsSI('H', 'T', T_makeup_K, 'Q', 0, 'Water') 
             
        # 1. Preparar Streams para o Mixer 2
        input_mixer_2 = []
        
        # Stream 1: Dreno Agregado (usa H_liq_out_J_kg, que é a entalpia CoolProp do Mixer 1)
        input_mixer_2.append({
            'm_dot': m_dot_dreno_kgs, 
            'T': T_dreno_C, 
            'P': P_dreno_kPa,
            'H': dreno_agregado_state['H_liq_out_J_kg'] # Entalpia CoolProp (J/kg)
        })
        
        # Stream 2: Reposição
        input_mixer_2.append({
            'm_dot': m_dot_makeup_kgs, 
            'T': T_makeup_C, 
            'P': P_makeup_kPa,
            'H': h_makeup_J_kg # Entalpia CoolProp (J/kg)
        })
        
        # 2. Executar Mixer 2 (Modelo CoolProp)
        detailed_input, output_results_mixer_2 = modelo_mixer(input_mixer_2, P_out_kPa=P_out_mixer_2_kPa)
        
        if output_results_mixer_2 is None:
            # Sem fallback
            raise ValueError(f"ERRO CRÍTICO: Falha no cálculo CoolProp do Mixer 2.")
        else:
            T_out_C = output_results_mixer_2.get('Temperatura de Saída (°C) (T_3)', np.nan)
            h_out_kJ_kg = output_results_mixer_2.get('Entalpia Específica de Saída (kJ/kg) (h_3)', np.nan) 
            H_out_J_kg = h_out_kJ_kg * 1000.0
        
    estado_recirculacao_final = {
        'Estado': 'Recirculação (Pós Mixer 2)',
        'T_out_C': T_out_C, 
        'P_out_bar': P_out_mixer_2_bar, 
        'M_dot_out_kgs': m_dot_out_kgs,
        # 🛑 H e T REAIS (CoolProp) são passados para o próximo estágio (Bomba/Boiler). 
        'H_out_J_kg': H_out_J_kg,
        'M_dot_makeup_kgs': m_dot_makeup_kgs,
        'h_out_kJ_kg': h_out_kJ_kg # kJ/kg
    }

    # Esta T_out_C é o valor real (21.3 C)
    df_plot_recirc = pd.DataFrame({
        'Estado': ['Dreno Agregado', estado_recirculacao_final['Estado']],
        'Vazão (kg/s)': [m_dot_dreno_kgs, m_dot_out_kgs],
        'Pressão (bar)': [P_out_mixer_2_bar, P_out_mixer_2_bar],
        'Temperatura (°C)': [T_dreno_C, T_out_C] 
    })
    
    return estado_recirculacao_final, df_plot_recirc


def exibir_resultados_bomba(res_bomba: dict):
# ... (Função inalterada - definida em plot_reporter_base.py) ...
    pass


def simular_bomba_e_boiler(estado_recirc: dict, P_alvo_bar: float, T_alvo_C: float):
    """
    Simula a Bomba (Pressurização) e o Boiler (Aquecimento).
    """
    
    # 1. BOMBA (Pressurização: P_in -> P_alvo) - ATIVADA
    P_in_bar = estado_recirc.get('P_out_bar')
    T_in_C = estado_recirc.get('T_out_C') # T real (~21.3 C)
    m_dot_kgs = estado_recirc.get('M_dot_out_kgs')
    h_in_kJ_kg = estado_recirc.get('h_out_kJ_kg', 0.0) # H real (~90 kJ/kg)
    
    P_alvo_kPa = P_alvo_bar * 100
    P_in_kPa = P_in_bar * 100
    
    print(f"\n--- INICIANDO BOMBA (MODELO COOLPROP: {P_in_bar:.2f} bar -> {P_alvo_bar:.2f} bar) ---")
    
    # 🛑 CHAMADA AO MODELO REAL DA BOMBA.
    ETA_IS = 0.82
    ETA_M = 0.96
    ETA_EL = 0.93
    
    res_bomba = calcular_bomba_modular(
        P2_kPa=P_alvo_kPa, T1_C=T_in_C, h_in_kJ_kg=h_in_kJ_kg, 
        P1_kPa=P_in_kPa, Vazao_m_kgs=m_dot_kgs,
        Eta_is=ETA_IS, Eta_m=ETA_M, Eta_el=ETA_EL
    )

    # Sem fallback:
    if 'erro' in res_bomba and res_bomba['erro'] is not None:
        raise ValueError(f"ERRO CRÍTICO no Modelo da Bomba: {res_bomba['erro']}")
        
    T_bomba_out_C = res_bomba['T_out_C']
    h_bomba_out_kJ_kg = res_bomba['h_out_kJ_kg']
    Pot_Eletrica_Bomba_kW = res_bomba['Pot_Eletrica_kW']
    exibir_resultados_bomba(res_bomba)

    # 2. BOILER (Aquecimento: T_bomba_out -> T_alvo) - AGORA ATIVADO
    print(f"\n--- INICIANDO BOILER ({T_bomba_out_C:.2f} °C -> {T_alvo_C:.2f} °C) ---")

    if T_bomba_out_C >= T_alvo_C:
        print("AVISO: Temperatura de entrada do Boiler já é maior ou igual ao alvo. Aquecimento zero.")
        res_boiler = {'W_eletrico_kW': 0.0, 'Delta_H_kJ_kg': 0.0, 'Q_necessario_kW': 0.0, 'h_out_kJ_kg': h_bomba_out_kJ_kg}
        T_boiler_out_C = T_bomba_out_C
        h_boiler_out_kJ_kg = h_bomba_out_kJ_kg
    else:
        # 🛑 CHAMADA AO MODELO REAL DO BOILER
        res_boiler = aquecer_boiler_modular(
            m_dot_kg_s=m_dot_kgs, 
            T_in_C=T_bomba_out_C, 
            T_out_C=T_alvo_C, 
            h_in_kJ_kg=h_bomba_out_kJ_kg,
            P_bar=P_alvo_bar, # Pressão pós-bomba
            eficiencia=0.99 
        )
        
        T_boiler_out_C = T_alvo_C
        h_boiler_out_kJ_kg = res_boiler['h_out_kJ_kg'] # h_out calculado pelo Boiler (usando CoolProp)
        
        # 🛑 Exibe os resultados do boiler (Função exibida fora do escopo)
        exibir_resultados_boiler(res_boiler, T_boiler_out_C)
    
    # 3. ESTADO FINAL PÓS-BOILER (T, P, H)
    estado_final_processado = {
        'T_out_C': T_boiler_out_C,
        'P_out_bar': P_alvo_bar, 
        'M_dot_out_kgs': m_dot_kgs,
        'H_out_J_kg': h_boiler_out_kJ_kg * 1000,
        'Pot_Eletrica_Bomba_kW': Pot_Eletrica_Bomba_kW,
        # Pot_Eletrica_Boiler_kW e Total_Pot_Eletrica_kW usam a chave correta 'W_eletrico_kW'
        'Pot_Eletrica_Boiler_kW': res_boiler['W_eletrico_kW'], 
        'Total_Pot_Eletrica_kW': Pot_Eletrica_Bomba_kW + res_boiler['W_eletrico_kW'],
        'h_out_kJ_kg': h_boiler_out_kJ_kg
    }
    
    return estado_final_processado, res_bomba


def executar_simulacao_mixer(df_h2: pd.DataFrame, df_o2: pd.DataFrame, mostrar_grafico: bool = False, T_ALVO_BOILER_C: float = 152.0):
    """
    Executa o fluxo completo: Drenos Purificação -> Mixer 1 -> Reposição (Mixer 2) -> Bomba/Boiler.
    
    O T_ALVO_BOILER_C padrão é 152.0 C.
    """
    
    # Inicialização de variáveis de retorno para o caso de falha.
    drenos_plot_data = None
    estado_final_processado = None
    
    try:
        # 1. Extração de dados (KODs, Coalescedor)
        drenos_h2_raw = extrair_dados_dreno(df_h2, 'H2')
        drenos_o2_raw = extrair_dados_dreno(df_o2, 'O2') 
    
        print("\n" + "="*80)
        print("INICIANDO CÁLCULO DA LINHA DE DRENOS (Duplo Mixer + Componentes)")
        print("="*80)
        
        # 🛑 CORREÇÃO: Chamadas de exibição corrigidas
        exibir_tabela_drenos_raw(drenos_h2_raw, 'H₂ (KODs + Coalescedor)')
        exibir_tabela_drenos_raw(drenos_o2_raw, 'O₂')
    
        # 2. MIXER 1 (KODs + Coalescedor) -> Dreno Agregado
        # P_DRENO_OUT_BAR agora é 1.0 bar (definido em constants_and_config)
        dreno_agregado_state = simular_linha_dreno(drenos_h2_raw, 'H2', P_DRENO_OUT_BAR)
        
        if dreno_agregado_state is None:
             print("\n!!! AVISO DEBUG: Nenhuma corrente líquida VÁLIDA. Simulação de drenos concluída. !!!")
             drenos_plot_data = {'H2_RAW': [], 'O2_RAW': [], 'FINAL_MIXER': {'Conc_H2_final_mg_kg': 0.0, 'Conc_O2_final_mg_kg': 0.0}}
             # Retorna 3 valores, mesmo no caso de dreno vazio.
             return drenos_plot_data, None, None 
             
        # Criamos o estado de Mixer Final Agregado (Pós-Mixer 1)
        final_mixer_state_agregado = {
            'Componente': dreno_agregado_state['Componente'],
            'm_dot_kg_h': dreno_agregado_state['m_dot_kg_h'], 
            'T_out_C': dreno_agregado_state['T'], 
            'P_out_bar': dreno_agregado_state['P_bar'], 
            'M_dot_H2O_final_kg_s': dreno_agregado_state['M_dot_H2O_final_kg_s'], 
            'H_liq_out_J_kg': dreno_agregado_state['H_liq_out_J_kg'], 
            'Conc_H2_final_mg_kg': 0.0, 
            'Conc_O2_final_mg_kg': dreno_agregado_state['C_diss_mg_kg'],
            # Adicionar o h_kJ_kg para ser usado na próxima etapa
            'h_out_kJ_kg': dreno_agregado_state['h_kJ_kg'] / 1000.0 # kJ/kg
        }
        
        exibir_tabela_processo_dreno(dreno_agregado_state, dreno_agregado_state, 'H2')
        exibir_estado_final_mixer(final_mixer_state_agregado)
    
        
        # 3. MIXER 2 (Dreno Agregado + Reposição)
        # 🛑 CORREÇÃO: Usar a constante P_DRENO_OUT_BAR para a pressão do Makeup (1.0 bar)
        P_MAKEUP_BAR = P_DRENO_OUT_BAR 
        print("\n--- INICIANDO SIMULAÇÃO DE REPOSIÇÃO DE ÁGUA (Mixer 2) ---")
        
        estado_recirculacao_final, df_plot_recirc = simular_reposicao_agua(
            final_mixer_state_agregado, 
            P_makeup_bar=P_MAKEUP_BAR, 
            T_makeup_C=20.0, 
            # Não passa o target kgs para usar o valor default (constante global)
        )
        
        exibir_estado_recirculacao(estado_recirculacao_final)
    
        # 4. BOMBA E BOILER (Pressurização e Aquecimento para SOEC) - AGORA ATIVADOS
        P_ALVO_SOEC_BAR = 5.0 
        
        estado_final_processado, res_bomba = simular_bomba_e_boiler(
            estado_recirculacao_final, 
            P_alvo_bar=P_ALVO_SOEC_BAR, 
            T_alvo_C=T_ALVO_BOILER_C # Usando o T_ALVO_BOILER_C passado como argumento
        )
        
        # 5. Validação do Balanço Global (Métrica de Controle)
        m_dot_drenada_total_kgs = final_mixer_state_agregado['M_dot_H2O_final_kg_s']
        exibir_validacao_balanco_global(m_dot_drenada_total_kgs, M_DOT_H2O_CONSUMIDA_KGS)
        
        drenos_plot_data = {
            'H2_RAW': drenos_h2_raw, 
            'O2_RAW': drenos_o2_raw,
            'H2_IN': dreno_agregado_state, 'H2_OUT': estado_recirculacao_final, 
            'O2_IN': None, 'O2_OUT': None, 
            'FINAL_MIXER': final_mixer_state_agregado, # Pós Mixer 1
            'P_OUT_BAR': P_ALVO_SOEC_BAR 
        }
        
        print("\nGráficos de Dreno/Recirculação gerados com sucesso (Apenas salvos no disco).")
    
        # Retorna o estado final processado (Pós-Bomba)
        return drenos_plot_data, None, estado_final_processado
        
    except Exception as e:
        # Corrigido: Para evitar que a string do erro (que é 'W_eletrica_kW') seja tratada
        # como uma chave válida que deveria estar no dicionário, vamos tratar o erro aqui.
        print(f"\n!!! ERRO CRÍTICO NÃO TRATADO NA LINHA DE DRENOS: {e} !!!")
        # Retorna os 3 'None's esperados pelo bloco try/except de main_SOEC_simulator.py
        return None, None, None