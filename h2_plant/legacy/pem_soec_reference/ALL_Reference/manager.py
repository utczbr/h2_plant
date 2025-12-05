import numpy as np
import soec_operator 
import pem_operator 
import matplotlib.pyplot as plt 
import report
import pandas as pd
import os

# --- SYSTEM CONSTANTS ---
MAX_PEM_POWER = 5.0 # MW (PEM Electrolyzer Limit)
NUM_SOEC_MODULES = 6
MAX_NOMINAL_MODULE_POWER = 2.4 # MW
OPTIMAL_LIMIT = 0.80 # 80%
MW_TON_H2 = 33.3 # MW per ton of H2 (Conversion Constant)
H2_PRICE_EUR_KG = 9.6 # Euros per kg of H2

# NOVO: Fatores de consumo extra de água
EXTRA_WATER_CONSUMPTION_SOEC = 0.10 # 10% de água não reagida consumida (vapor)
EXTRA_WATER_CONSUMPTION_PEM = 0.02  # 2% de água não reagida consumida

# NOVO: Constantes do SOEC (Importadas do operador ou usando fallback)
try:
    SOEC_H2_CONSUMPTION_KWH_PER_KG = soec_operator.BASE_H2_CONSUMPTION_KWH_PER_KG
    SOEC_STEAM_CONSUMPTION_KG_PER_MWH = soec_operator.BASE_STEAM_CONSUMPTION_KG_PER_MWH
except AttributeError:
    # Fallback para o caso de falha na importação do operador
    SOEC_H2_CONSUMPTION_KWH_PER_KG = 37.5
    SOEC_STEAM_CONSUMPTION_KG_PER_MWH = 672.0 / MAX_NOMINAL_MODULE_POWER # 280.0

# --- PRICE CONSTANTS (ARBITRARY) ---
PPA_PRICE_EUR_MWH = 50.0 # Fixed PPA contract price (EUR/MWh)

ROTATION_PERIOD_MINUTES = 60
STAND_BY_POWER = 0.0 # ou 0.05 se houver consumo mínimo
active_virtual_map = np.arange(NUM_SOEC_MODULES) # [0, 1, 2, 3, 4, 5]

# Estados para contagem de Ciclos (1=Standby, 2=Ramp/On)
prev_module_states = np.full(NUM_SOEC_MODULES, 1, dtype=int) 
total_cycle_counts = np.zeros(NUM_SOEC_MODULES, dtype=int)

# Histórico detalhado (Matriz para o relatório)
module_power_history_list = []

def distribute_power_among_modules(total_power_mw, virtual_map):
    """
    Distribui a potência Total (MW) entre os módulos baseando-se no mapa virtual.
    Replica a lógica simplificada do script de desgaste.
    """
    num_modules = len(virtual_map)
    module_powers_real = np.zeros(num_modules, dtype=float)
    
    # Constantes locais (baseadas no seu script de desgaste)
    OPTIMAL_LIMIT_MW = MAX_NOMINAL_MODULE_POWER * OPTIMAL_LIMIT # 1.92 MW
    
    # 1. Quantos módulos 'cheios' (N_floor) e sobra
    if total_power_mw <= 0.001:
        return module_powers_real # Tudo zero
        
    numerator = total_power_mw
    n_full = int(numerator // OPTIMAL_LIMIT_MW)
    remainder = numerator % OPTIMAL_LIMIT_MW
    
    # Lógica de teto (se a sobra for muito pequena, ignora ou ajusta)
    # Aqui usaremos uma lógica simplificada de preenchimento sequencial via Virtual Map
    
    # Vetor de potências VIRTUAIS
    powers_virtual = np.zeros(num_modules, dtype=float)
    
    # Preencher módulos virtuais sequencialmente
    for i in range(num_modules):
        if total_power_mw >= OPTIMAL_LIMIT_MW:
            powers_virtual[i] = OPTIMAL_LIMIT_MW
            total_power_mw -= OPTIMAL_LIMIT_MW
        else:
            powers_virtual[i] = total_power_mw
            total_power_mw = 0.0
            break
            
    # Mapear de volta para REAL (Usando o índice do virtual map)
    # Se virtual_map[0] = 3 (Módulo Real 3 é o 1º Virtual) -> module_powers_real[3] = powers_virtual[0]
    for v_idx, real_idx in enumerate(virtual_map):
        module_powers_real[real_idx] = powers_virtual[v_idx]
        
    return module_powers_real

def carregar_dados_reais_interpolados(horas_simulacao=24):
    """
    Carrega CSVs de Vento e Preço, converte para NUMÉRICO (Float) 
    e interpola para resolução de 1 minuto.
    """
    print("--- Carregando dados reais dos CSVs ---")
    
    # --- 1. CARREGAR PREÇOS ---
    try:
        # Lê usando primeira coluna como índice
        df_precos = pd.read_csv('NL_Prices_2024_15min.csv', index_col=0)
        
        # Converte índice para Datetime e remove Timezone
        df_precos.index = pd.to_datetime(df_precos.index, utc=True).tz_localize(None)
        df_precos.sort_index(inplace=True)
        
        # Identifica coluna de preço
        col_valor_preco = [c for c in df_precos.columns if 'price' in c.lower() or 'eur' in c.lower()][0]
        
        # LIMPEZA DE DADOS (CRUCIAL): Troca vírgula por ponto e converte para Float
        if df_precos[col_valor_preco].dtype == 'object':
            df_precos[col_valor_preco] = df_precos[col_valor_preco].astype(str).str.replace(',', '.')
        
        df_precos[col_valor_preco] = pd.to_numeric(df_precos[col_valor_preco], errors='coerce')
        
    except Exception as e:
        print(f"ERRO ao ler arquivo de preços: {e}")
        return [], []

    # --- 2. CARREGAR VENTO ---
    try:
        # Tenta ler (padrão ponto decimal)
        df_vento = pd.read_csv('producao_horaria_2_turbinas.csv', sep=',', decimal='.')
        
        # Fallback (padrão vírgula decimal)
        if len(df_vento.columns) < 2:
            df_vento = pd.read_csv('producao_horaria_2_turbinas.csv', sep=';', decimal=',')

        # Converte Data
        df_vento['data_hora'] = pd.to_datetime(df_vento['data_hora'])
        df_vento = df_vento.set_index('data_hora').sort_index()

        # LIMPEZA DE DADOS (CRUCIAL): Garante que Potência é Float
        col_vento = 'potencia_2_turbinas_MW'
        if df_vento[col_vento].dtype == 'object':
            df_vento[col_vento] = df_vento[col_vento].astype(str).str.replace(',', '.')
            
        df_vento[col_vento] = pd.to_numeric(df_vento[col_vento], errors='coerce')
        
    except Exception as e:
        print(f"ERRO ao ler arquivo de vento: {e}")
        return [], []

    # --- 3. FILTRAGEM E SINCRONIZAÇÃO ---
    if df_precos.empty or df_vento.empty:
        print("ERRO: DataFrames vazios após leitura.")
        return [], []

    data_inicio = df_precos.index[0]
    data_fim = data_inicio + pd.Timedelta(hours=horas_simulacao)
    
    # Recorte temporal
    df_precos = df_precos[(df_precos.index >= data_inicio) & (df_precos.index < data_fim)]
    df_vento = df_vento[(df_vento.index >= data_inicio) & (df_vento.index < data_fim)]
    
    # --- 4. UPSAMPLING (1 MINUTO) ---
    idx_minuto = pd.date_range(start=data_inicio, end=data_fim, freq='1min', inclusive='left')
    
    # Preços: Forward Fill
    series_preco_1min = df_precos[col_valor_preco].reindex(idx_minuto).ffill().bfill()
    
    # Vento: Interpolação Linear (Agora funciona pois os dados são floats!)
    # .infer_objects(copy=False) silencia o warning antigo
    series_vento_1min = df_vento[col_vento].reindex(idx_minuto).infer_objects(copy=False).interpolate(method='linear').ffill().bfill()
    
    print(f"Dados carregados: {len(idx_minuto)} minutos simulados.")
    print(f"Médias -> Preço: {series_preco_1min.mean():.2f} EUR | Vento: {series_vento_1min.mean():.2f} MW")
    
    return series_preco_1min.values, series_vento_1min.values

# Spot Price Profile (EUR/MWh) - 8 Hours, 4 periods of 15 minutes per hour. 32 values.
SPOT_PRICE_HOUR_BY_HOUR = [
    [40.0, 42.0, 45.0, 40.0], # H1 (Expected sale from 45-59, if SOEC surplus exists)
    [35.0, 35.0, 65.0, 55.0], # H2 (Expected sale from 60-119)
    [350.0, 380.0, 400.0, 420.0], # H3
    [500.0, 520.0, 505.0, 580.0], # H4
    [450.0, 480.0, 500.0, 520.0], # H5
    [55.0, 58.0, 60.0, 62.0], # H6
    [40.0, 42.0, 45.0, 48.0], # H7
    [30.0, 32.0, 35.0, 38.0]  # H8
]

# Create the 480-minute spot price reference vector
SPOT_PRICE_PROFILE = []
for hour in SPOT_PRICE_HOUR_BY_HOUR:
    for price_15min in hour:
        SPOT_PRICE_PROFILE.extend([price_15min] * 15) # 15 minutes per price

# --- SOEC CONFIGURATIONS ---
REAL_OFF_MODULES = [] 
ROTATION_ACTIVE = False        
EFFICIENT_POWER_LIMIT = True 

# Maximum Capacity Calculations
if EFFICIENT_POWER_LIMIT:
    MAX_SOEC_MODULE_POWER = MAX_NOMINAL_MODULE_POWER * OPTIMAL_LIMIT # 1.92 MW
else:
    MAX_SOEC_MODULE_POWER = MAX_NOMINAL_MODULE_POWER # 2.4 MW
    
SOEC_MAX_CAPACITY = MAX_SOEC_MODULE_POWER * NUM_SOEC_MODULES # 6 * 1.92 = 11.52 MW

# --- NEW OFFERED POWER PROFILE (8 hours, 480 minutes) ---
HOUR_OFFER = [3.0, 5.0, 13.0, 18.0, 15.0, 9.0, 18.0, 0.0] # MW
# Create the 480-minute reference vector
OFFERED_POWER_PROFILE = []
for pot in HOUR_OFFER:
    OFFERED_POWER_PROFILE.extend([pot] * 60) # 60 minutes per hour

# --- STATE VARIABLE INITIALIZATION ---

# Inicialização do estado PEM
try:
    initial_pem_state = pem_operator.initialize_pem_simulation()
    pem_state = initial_pem_state
except AttributeError:
    # Stub simples para PemState
    class PemStateStub:
        def __init__(self):
            self.t_op_h = 0.0
            self.H_EOL = 999999.0 
    pem_state = PemStateStub()
    # Stub para a função PemStep
    def run_pem_step_stub(P_target_kW, current_state):
        current_state.t_op_h += 1.0 / 60.0 
        # Cálculo de produção aproximado: 33.3 kWh/kg -> 30.03 kg/MWh
        h2_rate = 30.03 # kg/MWh
        
        m_H2_kg = P_target_kW / 1000.0 / 60.0 * h2_rate
        m_O2_kg = m_H2_kg * 8.0 # H2:O2 ratio is 1:8 mass
        m_H2O_kg = m_H2_kg * 9.0 # H2:H2O ratio is 1:9 mass
        
        return m_H2_kg, m_O2_kg, m_H2O_kg, current_state
        
    pem_operator.run_pem_step = run_pem_step_stub
    pem_operator.PemState = PemStateStub
    
# Inicialização do estado SOEC
initial_soec_state = soec_operator.initialize_soec_simulation(
    off_real_modules=REAL_OFF_MODULES, 
    rotation_enabled=ROTATION_ACTIVE,
    use_optimal_limit=EFFICIENT_POWER_LIMIT,
    year=0.0 # Pode parametrizar isso se quiser simular anos futuros
)
soec_state = initial_soec_state


PREVIOUS_SOEC_POWER = 0.0 
CURRENT_PEM_POWER = 0.0
CURRENT_SOLD_POWER = 0.0
try:
    NUM_ACTIVE_MODULES = soec_operator.NUM_ACTIVE_MODULES 
except AttributeError:
    NUM_ACTIVE_MODULES = NUM_SOEC_MODULES 
FORCE_SELL_FLAG = False # Main flag for total SOEC BYPASS

# --- SIMULATION HISTORY ---
history = {
    'minute': [], 'hour': [], 'P_offer': [], 'P_soec_set': [], 
    'P_soec_actual': [], 'P_pem': [], 'P_sold': [], 'P_previous': [],
    'spot_price': [], 'sell_decision': [], 'H2_soec_kg': [], 
    'steam_soec_kg': [], 
    'H2_pem_kg': [], 
    'O2_pem_kg': [], 
    'H2O_pem_kg': [] 
}

# FUNÇÃO AUXILIAR: Cálculo de consumo total de água por passo
def calculate_total_water_consumed_per_step(history):
    """Calcula o consumo total de água (reação + extra) para SOEC e PEM em kg/min."""
    
    # SOEC: Steam Consumed (H2O de reação) + 10% extra.
    # Assumimos que o vapor (steam_soec_kg) é o consumo de H2O de reação
    water_soec_reaction_kg = np.array(history['steam_soec_kg'])
    total_water_soec_kg = water_soec_reaction_kg * (1 + EXTRA_WATER_CONSUMPTION_SOEC)
    
    # PEM: H2O Consumed (H2O de reação) + 2% extra.
    water_pem_reaction_kg = np.array(history['H2O_pem_kg'])
    total_water_pem_kg = water_pem_reaction_kg * (1 + EXTRA_WATER_CONSUMPTION_PEM)
    
    total_water_system_kg = total_water_soec_kg + total_water_pem_kg
    
    return total_water_system_kg, total_water_soec_kg, total_water_pem_kg

# FUNÇÃO DE LOG
def log_history(minute, P_offer, P_soec_set, P_soec_actual, P_pem, P_sold, P_previous, spot_price, sell_decision, h2_soec_kg, steam_soec_kg, h2_pem_kg, o2_pem_kg, h2o_pem_kg):
    """Logs the current state of the simulation."""
    history['minute'].append(minute)
    history['hour'].append(minute // 60 + 1)
    history['P_offer'].append(P_offer)
    history['P_soec_set'].append(P_soec_set)
    history['P_soec_actual'].append(P_soec_actual)
    history['P_pem'].append(P_pem)
    history['P_sold'].append(P_sold)
    history['P_previous'].append(P_previous)
    history['spot_price'].append(spot_price)
    history['sell_decision'].append(sell_decision)
    history['H2_soec_kg'].append(h2_soec_kg) 
    history['steam_soec_kg'].append(steam_soec_kg) 
    history['H2_pem_kg'].append(h2_pem_kg) 
    history['O2_pem_kg'].append(o2_pem_kg) 
    history['H2O_pem_kg'].append(h2o_pem_kg) 

# FUNÇÃO DE DECISÃO E DESPACHO
def decide_and_execute_dispatch(minute, P_offer, P_future_offer, P_soec_previous, current_soec_state: soec_operator.SoecState, current_pem_state: pem_operator.PemState):
    """Hybrid Manager's decision criterion for the current minute."""
    global CURRENT_PEM_POWER
    global CURRENT_SOLD_POWER
    global ROTATION_ACTIVE
    global SOEC_MAX_CAPACITY
    global FORCE_SELL_FLAG 
    global SPOT_PRICE_PROFILE
    global PPA_PRICE_EUR_MWH
    global MW_TON_H2
    global H2_PRICE_EUR_KG
    global SOEC_H2_CONSUMPTION_KWH_PER_KG 
    global SOEC_STEAM_CONSUMPTION_KG_PER_MWH 
    
    P_soec_set = 0.0
    P_soec_actual = 0.0
    P_pem = 0.0
    P_sold = 0.0
    sell_decision = 0 
    h2_soec_kg = 0.0 
    steam_soec_kg = 0.0 
    
    # NOVAS SAÍDAS PEM
    h2_pem_kg = 0.0
    o2_pem_kg = 0.0
    h2o_pem_kg = 0.0
    
    minute_of_hour = minute % 60
    
    # Current minute Spot Price
    current_spot_price = SPOT_PRICE_PROFILE[minute]
    
    # Recalculate Arbitrage Limit (H2 equivalent price based on SOEC efficiency)
    h2_eq_price = (1000.0 / SOEC_H2_CONSUMPTION_KWH_PER_KG) * H2_PRICE_EUR_KG # EUR/MWh
    arbitrage_limit = PPA_PRICE_EUR_MWH + h2_eq_price

    # 1. Initial Check
    offer_previous_difference = P_offer - P_soec_previous

    # --- ARBITRAGE STEP (ONLY at minute 0 of the hour AND with RAMP UP) ---
    if minute_of_hour == 0 and offer_previous_difference > 0.0: # Ramp Up
        
        P_surplus = P_offer - P_soec_previous 
        time_h_arbitrage = 15.0 / 60.0
        
        # Sales Profit
        sale_profit = P_surplus * time_h_arbitrage * (current_spot_price - PPA_PRICE_EUR_MWH)
        
        # H2 Profit (Calculated using SOEC efficiency for consistency)
        E_surplus_mwh = P_surplus * time_h_arbitrage 
        mass_kg = E_surplus_mwh * (1000.0 / SOEC_H2_CONSUMPTION_KWH_PER_KG)
        h2_profit = mass_kg * H2_PRICE_EUR_KG
        
        # Decision
        if sale_profit > h2_profit:
            FORCE_SELL_FLAG = True
        else:
            FORCE_SELL_FLAG = False

    # --- ARBITRAGE CONTROL BLOCK (Continuous) ---

    # 1. Check for End of Economic Advantage and Reset Flag (Continuous)
    if FORCE_SELL_FLAG and (current_spot_price <= arbitrage_limit):
         FORCE_SELL_FLAG = False

    # 2. Reset due to Future Ramp Down (Minute 45) - OPERATIONAL PRIORITY (CORRIGIDO)
    if minute_of_hour == 45:
         
         # 1. Calcular o Set Point Máximo que o SOEC pode ter na próxima hora (H5)
         P_soec_set_fut = min(P_future_offer, SOEC_MAX_CAPACITY)
         
         # 2. Resetar o flag APENAS se a potência anterior do SOEC (P_soec_previous) 
         # for MAIOR do que o que o SOEC pode consumir na próxima hora (indicando um Ramp Down operacional necessário).
         if P_soec_previous > P_soec_set_fut:
             FORCE_SELL_FLAG = False
    
    # 3. Re-evaluation: Sale is active if FORCE_SELL_FLAG is True
    sell_decision = 1 if FORCE_SELL_FLAG else 0 

    # Initialize output state
    updated_soec_state = current_soec_state
    updated_pem_state = current_pem_state
    
    
    if sell_decision == 1: # Priority 1: TOTAL SALE BYPASS OF SOEC
        
        P_soec_set = P_soec_previous 
        P_soec_actual = P_soec_previous
        P_pem = 0.0
        
        soec_offer_difference = P_offer - P_soec_actual 
        P_sold = soec_offer_difference 
        
        # Stub calculation for H2/Steam when bypassing
        try:
            if P_soec_actual > 0.0:
                energy_consumed_mwh = P_soec_actual / 60.0
                h2_rate = 1000.0 / SOEC_H2_CONSUMPTION_KWH_PER_KG 
                steam_rate = SOEC_STEAM_CONSUMPTION_KG_PER_MWH 

                h2_soec_kg = energy_consumed_mwh * h2_rate
                steam_soec_kg = energy_consumed_mwh * steam_rate 
        except (AttributeError, NameError):
            h2_soec_kg = 0.0 
            steam_soec_kg = 0.0 
            
        # PEM Bypassed/Zeroed
        h2_pem_kg = 0.0
        o2_pem_kg = 0.0
        h2o_pem_kg = 0.0
        
    else: 
        # --- Normal H2 Production Logic (SOEC and PEM) ---
        
        # 1. SOEC Set Point Calculation 
        P_soec_set_real = P_offer
        if P_offer > SOEC_MAX_CAPACITY:
            P_soec_set_real = SOEC_MAX_CAPACITY
        
        if minute_of_hour >= 45 and minute_of_hour < 60:
            future_offer_difference = P_future_offer - P_offer
            if future_offer_difference < 0:
                P_soec_set_fut = min(P_future_offer, SOEC_MAX_CAPACITY)
                P_soec_set_real = min(P_soec_set_real, P_soec_set_fut)
            
        P_soec_set = P_soec_set_real

        # 2. Execute step with adjusted Set Point (call the black box SOEC)
        try:
            P_soec_actual, updated_soec_state, h2_soec_kg, steam_soec_kg = soec_operator.run_soec_step(
                P_soec_set, current_soec_state
            )
        except AttributeError:
             P_soec_actual = min(P_soec_set, SOEC_MAX_CAPACITY)
             updated_soec_state = current_soec_state 
             h2_soec_kg = 0.0 
             steam_soec_kg = 0.0 

        # 3. PEM AND SOLD DISPATCH 
        soec_offer_difference = P_offer - P_soec_actual # Surplus after SOEC
        
        PEM_SELL_FLAG_LOCAL = False
        
        # Arbitrage (Saturated SOEC/Surplus)
        if soec_offer_difference > 0.0 and current_spot_price > arbitrage_limit:
            PEM_SELL_FLAG_LOCAL = True
            
        # Priority 1: Sell Surplus (PEM Bypass) if Advantageous (ANY MINUTE)
        if PEM_SELL_FLAG_LOCAL:
             P_sold = soec_offer_difference
             P_pem = 0.0
        
        # Priority 2: H2 Production (PEM) if not selling
        elif soec_offer_difference > 0.0:
             # Potência para PEM (MW)
             P_pem_alloc_MW = min(soec_offer_difference, MAX_PEM_POWER)
             P_pem = P_pem_alloc_MW
             P_sold = soec_offer_difference - P_pem
             
             # NOVO: Executar passo PEM
             P_pem_alloc_kW = P_pem_alloc_MW * 1000.0
             
             # CHAMADA CORRIGIDA
             h2_pem_kg, o2_pem_kg, h2o_pem_kg, updated_pem_state = pem_operator.run_pem_step(
                 P_pem_alloc_kW, # Potência em kW
                 current_pem_state
             )
        else:
             P_pem = 0.0
             h2_pem_kg = 0.0
             o2_pem_kg = 0.0
             h2o_pem_kg = 0.0
             updated_pem_state = current_pem_state

        # 4. Update sell_decision for History/Graph
        if sell_decision == 0 and PEM_SELL_FLAG_LOCAL:
             sell_decision = 1 
    
    # GLOBAL STATE UPDATE
    CURRENT_PEM_POWER = P_pem
    CURRENT_SOLD_POWER = P_sold

    # RETORNO ATUALIZADO: Adicionado updated_pem_state e todas as saídas de massa
    return (
        updated_soec_state, updated_pem_state,
        P_soec_actual, P_soec_set, P_pem, P_sold, current_spot_price, sell_decision, 
        h2_soec_kg, steam_soec_kg, h2_pem_kg, o2_pem_kg, h2o_pem_kg
    )


def run_hybrid_management():
    """Main function to run the hybrid management simulation with arbitrage."""
    global soec_state, pem_state, PREVIOUS_SOEC_POWER,OFFERED_POWER_PROFILE

    
    
    #times = len(HOUR_OFFER) * 60 # 8 * 60 = 480 minutes
    # --- NOVO: Carregar dados reais ---
    HORAS_PARA_SIMULAR = 24*360 # Exemplo:
    vec_precos, vec_vento = carregar_dados_reais_interpolados(HORAS_PARA_SIMULAR)
    
    OFFERED_POWER_PROFILE = vec_vento

    if len(vec_precos) == 0:
        return # Aborta se falhou o load
        
    # Substitui a variável global SPOT_PRICE_PROFILE pelo vetor real carregado
    global SPOT_PRICE_PROFILE 
    SPOT_PRICE_PROFILE = vec_precos
    
    times = len(vec_precos)

    h2_eq_price = (1000/SOEC_H2_CONSUMPTION_KWH_PER_KG) * H2_PRICE_EUR_KG 
    arbitrage_limit = PPA_PRICE_EUR_MWH + h2_eq_price

    active_virtual_map = np.arange(NUM_SOEC_MODULES) # [0, 1, 2, 3, 4, 5]
    prev_module_states = np.full(NUM_SOEC_MODULES, 1, dtype=int) 
    total_cycle_counts = np.zeros(NUM_SOEC_MODULES, dtype=int)
    module_power_history_list = [] 
    
    # Constante de rotação (se não estiver definida globalmente)
    ROTATION_PERIOD_MINUTES = 60
    
    print("\n--- Starting Hybrid Management Simulation with Arbitrage (8 Hours / 480 Minutes) ---")
    print(f"Max SOEC Capacity (80%): {SOEC_MAX_CAPACITY:.2f} MW")
    print(f"Max PEM Capacity: {MAX_PEM_POWER:.2f} MW")
    print(f"Hydrogen Price: {H2_PRICE_EUR_KG:.2f} EUR/kg | PPA Price: {PPA_PRICE_EUR_MWH:.2f} EUR/MWh")
    print(f"SOEC H2 Efficiency: {SOEC_H2_CONSUMPTION_KWH_PER_KG:.2f} kWh/kg H2")
    print(f"SOEC Steam Consumption: {SOEC_STEAM_CONSUMPTION_KG_PER_MWH:.2f} kg/MWh")
    print(f"Arbitrage Point (Sale > H2): > {arbitrage_limit:.2f} EUR/MWh")
    print("----------------------------------------------------------------------------------------------------------------------\n")

    # Detailed Print Table - ATUALIZADO: Adicionado H2 PEM e H2O PEM
    print(" | Min | H | P. Offer | P. SOEC | P. PEM | P. Sold | Spot | Dec | H2 SOEC (kg/min) | H2 PEM (kg/min) | H2O PEM (kg/min) |")
    print(" |-----|---|-----------|---------|--------|---------|------|-----|------------------|-----------------|------------------|")


    for minute in range(times):
        
        P_offer = OFFERED_POWER_PROFILE[minute]
        
        # Future Power 
        if minute + 60 < times:
             P_future_offer = OFFERED_POWER_PROFILE[minute + 60]
        else:
            P_future_offer = P_offer 
        
        P_soec_previous = PREVIOUS_SOEC_POWER # P(t-1)
        
        P_soec_previous = PREVIOUS_SOEC_POWER # P(t-1)
        
        # CHAMADA ATUALIZADA: Passa o estado PEM e recebe 13 valores
        (
            updated_soec_state, updated_pem_state, 
            P_soec_actual, P_soec_set, P_pem, P_sold, current_spot_price, sell_decision, 
            h2_soec_kg, steam_soec_kg, h2_pem_kg, o2_pem_kg, h2o_pem_kg
        ) = decide_and_execute_dispatch(
            minute, P_offer, P_future_offer, P_soec_previous, soec_state, pem_state
        )
        
        # Update state for the next step (t+1)
        soec_state = updated_soec_state
        pem_state = updated_pem_state # NOVO: Atualiza o estado PEM
        PREVIOUS_SOEC_POWER = P_soec_actual 

        # 1. Rotação do Mapa Virtual (a cada hora)
        if minute > 0 and minute % ROTATION_PERIOD_MINUTES == 0:
            active_virtual_map = np.roll(active_virtual_map, -1)
            
        # 2. Calcular potência individual de cada módulo
        current_module_powers = distribute_power_among_modules(P_soec_actual, active_virtual_map)
        module_power_history_list.append(current_module_powers)
        
        # 3. Detectar Ciclos (Startups)
        # Consideramos "Ligado" se Potência > 0.05 (Standby)
        current_states = np.where(current_module_powers > 0.05, 2, 1) # 2=On, 1=Off/Standby
        
        for i in range(NUM_SOEC_MODULES):
            # Transição 1 -> 2 (Off para On) conta como ciclo
            if prev_module_states[i] == 1 and current_states[i] == 2:
                total_cycle_counts[i] += 1
                
        prev_module_states = current_states
        
        # Log History - ATUALIZADO: Adicionado h2_pem_kg, o2_pem_kg, h2o_pem_kg
        log_history(minute, P_offer, P_soec_set, P_soec_actual, P_pem, P_sold, P_soec_previous, current_spot_price, sell_decision, h2_soec_kg, steam_soec_kg, h2_pem_kg, o2_pem_kg, h2o_pem_kg)

        if minute % 60 == 0:
            print(f"\n--- START OF HOUR {minute//60 + 1} --- (P. Offer: {P_offer:.2f} MW) ---")
        
        # Detailed printing for minutes 0, 15, 30, 45 of each hour
        if minute % 15 == 0:
             # PRINT ATUALIZADO: Simplificado e Adicionado H2 PEM e H2O PEM
             print(
                f" | {minute:03d} | {minute//60 + 1} | {P_offer:9.2f} | {P_soec_actual:7.2f} | {P_pem:6.2f} | {P_sold:7.2f} | {current_spot_price:5.0f} | {('SELL' if sell_decision == 1 else 'H2'):3s} | {h2_soec_kg:16.4f} | {h2_pem_kg:15.4f} | {h2o_pem_kg:16.4f} |"
            )

    print("----------------------------------------------------------------------------------------------------------------------")
    print("\n--- End of Hybrid Management Simulation ---")
   
    # Display summary
    print("\n## Simulation Summary (Total/Average Values)")
    E_total_offer = np.sum(history['P_offer']) / 60
    E_soec = np.sum(history['P_soec_actual']) / 60
    E_pem = np.sum(history['P_pem']) / 60
    E_sold = np.sum(history['P_sold']) / 60
    
    # H2/VAPOR/ÁGUA TOTAL CALCULATION
    H2_soec_total = np.sum(history['H2_soec_kg'])
    Steam_soec_reaction_total = np.sum(history['steam_soec_kg'])
    H2_pem_total = np.sum(history['H2_pem_kg']) 
    O2_pem_total = np.sum(history['O2_pem_kg']) 
    H2O_pem_reaction_total = np.sum(history['H2O_pem_kg']) 
    H2_total = H2_soec_total + H2_pem_total 

    # NOVO: Cálculo do consumo total de água (reação + extra)
    total_water_system_rate, total_water_soec_rate, total_water_pem_rate = calculate_total_water_consumed_per_step(history)
    Total_Water_System_Consumption = np.sum(total_water_system_rate)
    Total_Water_SOEC_Consumption = np.sum(total_water_soec_rate)
    Total_Water_PEM_Consumption = np.sum(total_water_pem_rate)

    print(f"* Total Offered Energy: {E_total_offer:.2f} MWh")
    print(f"* Energy Supplied to SOEC (H2/Steam): {E_soec:.2f} MWh")
    print(f"* Energy Supplied to PEM (H2/O2/H2O): {E_pem:.2f} MWh") 
    print(f"* **Total System Hydrogen Production**: {H2_total:.2f} kg") 
    print(f"  * SOEC Production: {H2_soec_total:.2f} kg")
    print(f"  * PEM Production: {H2_pem_total:.2f} kg")
    print(f"* **Total SOEC Steam Consumption (Reaction)**: {Steam_soec_reaction_total:.2f} kg") 
    print(f"* **Total SOEC Water Consumption (Total)**: {Total_Water_SOEC_Consumption:.2f} kg ({EXTRA_WATER_CONSUMPTION_SOEC*100:.0f}% Extra)")
    print(f"* **Total PEM Water Consumption (Reaction)**: {H2O_pem_reaction_total:.2f} kg")
    print(f"* **Total PEM Water Consumption (Total)**: {Total_Water_PEM_Consumption:.2f} kg ({EXTRA_WATER_CONSUMPTION_PEM*100:.0f}% Extra)")
    print(f"* **Total System Water Consumption**: {Total_Water_System_Consumption:.2f} kg") 
    print(f"* **Total PEM Oxygen Production**: {O2_pem_total:.2f} kg") 
    print(f"* Energy Sold to the Market: {E_sold:.2f} MWh")
    print(f"* Offer Deviation (Error Margin): {E_total_offer - (E_soec + E_pem + E_sold):.4f} MWh")
    print("-------------------------------------------------------------------")
    
    # Generate charts
    graph_requests = ['all'] 
    try:
        import report
        # Converter lista para array numpy para facilitar plotagem
        module_history_array = np.array(module_power_history_list)
        
        report.generate_selected_reports(
            history, 
            pem_state, 
            selection=graph_requests,
            module_history=module_history_array, # Passa matriz de potências
            module_cycles=total_cycle_counts     # Passa vetor de ciclos
        )
    except ImportError as e:
        print(f"ERRO CRÍTICO: Não foi possível importar o módulo 'report.py'. Detalhes: {e}")
    except Exception as e:
        print(f"ERRO durante a geração de relatórios: {e}")
        
    print(f"\nCharts saved as: hybrid_dispatch_arbitrage.png, arbitrage_prices_chart.png, total_energy_pie_chart.png, total_h2_production_chart.png, soec_reaction_water_consumption_chart.png and total_water_consumption_chart.png")


if __name__ == "__main__":
    run_hybrid_management()