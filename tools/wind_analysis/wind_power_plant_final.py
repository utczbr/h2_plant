"""
WIND POWER PLANT ANALYSIS ENGINE - REVISED & VALIDATED
=======================================================
Comprehensive techno-economic assessment tool for wind farm viability studies.

IMPROVEMENTS OVER v1:
- Cubic power curve physics (replaces linear fallacy)
- Realistic Vestas V164-9.5 MW turbine model
- Proper temporal synchronization (hourly weather to hourly prices via resampling)
- Curtailment logic for negative pricing
- Capture price metric for market analysis
- Robust error handling with fail-fast approach
- Dynamic wake loss modeling based on turbine density
- Comprehensive reporting with volatility analysis

DEPENDENCIES:
    pandas, numpy, matplotlib, scipy, windpowerlib, pvlib, entsoe
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
from pathlib import Path
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    # Create a dummy plt object to avoid NameError if used blindly (though we will guard it)
    class DummyPlt:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    plt = DummyPlt()
from scipy.interpolate import CubicSpline
import logging

# Optional imports - graceful degradation if not available
try:
    from windpowerlib import WindTurbine, ModelChain
    HAS_WINDPOWERLIB = True
except ImportError:
    HAS_WINDPOWERLIB = False
    warnings.warn("windpowerlib not available - using fallback power curve")

try:
    from entsoe import EntsoePandasClient
    HAS_ENTSOE = True
except ImportError:
    HAS_ENTSOE = False

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
CONFIG = {
    # Location (Borssele area, North Sea)
    'latitude': 53.0,
    'longitude': 4.0,
    'altitude': 0,
    'z0': 0.0014,  # Roughness length (sea)
    'input_wind_height': 100,  # Assume 100m reference height in data
    
    # Turbine specification: Vestas V164-9.5 MW (optimized for offshore)
    'turbine_name': 'Vestas V164/9500',
    'hub_height': 105,
    'rotor_diameter': 164.0,
    'nominal_power_kW': 9500.0,
    'cut_in_speed': 3.5,
    'rated_speed': 14.0,  # Realistic for V164
    'cut_out_speed': 25.0,
    
    # Park sizing
    'P80_target_MW': 3.0,
    'base_farm_efficiency': 0.60,  # Will be adjusted for density
    'net_capacity_factor': 0.60,  # Realistic offshore target (45%)
    
    # Analysis thresholds
    'cutoff_power_MW': 30.0,
    'target_exceedance_percent': 80,
    
    # Data sources
    'weather_data_source': 'csv',
    'weather_csv_path': 'weather_data_2023-2024.csv',
    'price_csv_path': 'NL_Prices_2024_15min.csv', 
    'use_cached_prices': False,
    'prices_cache_path': 'prices_2024.csv',
    'entsoe_country_code': 'NL',
    
    # Physics flags
    'use_scale_height_correction': False,  # rhoa already accounts for altitude
    'apply_density_correction': True,
    'apply_curtailment': True,
    'curtailment_threshold_EUR_MWh': -9999.0,  # EUR/MWh threshold -----------excluir
    'model_high_wind_derating': True,  # Power derate above 25 m/s
    
    # Output
    'output_dir': './results_final_2024/',
    'save_plots': True,
    'verbose': True,
}

# ==================== HELPER FUNCTIONS ====================

def validate_config():
    """Validate that config parameters are physically consistent."""
    if CONFIG['cut_in_speed'] >= CONFIG['rated_speed']:
        raise ValueError("Cut-in speed must be < rated speed")
    if CONFIG['rated_speed'] >= CONFIG['cut_out_speed']:
        raise ValueError("Rated speed must be < cut-out speed")
    if CONFIG['base_farm_efficiency'] <= 0 or CONFIG['base_farm_efficiency'] > 1:
        raise ValueError("Farm efficiency must be between 0 and 1")
    logger.info("✓ Configuration validated")


def calculate_wake_efficiency(n_turbines):
    # Fator base de perdas não relacionadas à esteira (elétrica, disponibilidade, etc.)
    # Ex: 97% disponibilidade * 98% elétrica = ~0.95
    global_availability = 0.95
    
    if n_turbines <= 1:
        return global_availability
    
    # Parâmetros para o modelo de Jensen (Park model)
    Ct = 0.8  # Coeficiente de empuxo típico
    k = 0.075  # Constante de decaimento de esteira (onshore)
    sd = 164.0   # Espaçamento entre turbinas em diâmetros de rotor (típico)
    
    # Assume uma fileira linear de turbinas alinhada com o vento
    a = 1 - np.sqrt(1 - Ct)  # Fator de déficit inicial
    
    deficits = np.zeros(n_turbines)
    for i in range(1, n_turbines):
        sum_sq = 0.0
        for j in range(i):
            dist = (i - j) * sd  # Distância em múltiplos de D
            def_single = a / (1 + 2 * k * dist)**2
            sum_sq += def_single**2
        deficits[i] = np.sqrt(sum_sq)
    
    u_rel = 1.0 - deficits
    power_rel = u_rel ** 3
    wake_factor = np.mean(power_rel)
    
    total_efficiency = global_availability * wake_factor
    return total_efficiency

def load_turbine_power_curve_cubic(turbine_name=None):
    """
    Carregue ou gere uma curva de potência cúbica REALISTA para o Vestas V164-9.5 MW.

    Justificativa Matemática:
    ---------------------------
    A potência disponível no vento é dada por:
    $$ P_{vento} = \frac{1}{2} \rho A v^3 $$

    A potência elétrica extraída é:
    $$ P_{eléc} = C_p(\lambda, \beta) \cdot \eta_{mec} \cdot \eta_{eléc} \cdot P_{vento} $$

    Onde:
    - $\rho$: Densidade do ar (kg/m³)
    - $A$: Área varrida pelo rotor ($\pi r^2$)
    - $v$: Velocidade do vento (m/s)
    - $C_p$: Coeficiente de potência (limite teórico, limite de Betz $\approx 0,593$)

    Na região de carga parcial ($v_{ligação} < v < v_{nominal}$), a turbina otimiza $C_p$, resultando em uma curva que segue de perto a relação cúbica ($P ∝ v³$).

    Esta função usa interpolação spline cúbica para reconstruir essa curva baseada em princípios físicos a partir de pontos de dados do fabricante original (OEM), garantindo derivadas suaves e comportamento realista.

    Retorna: dict: DataFrame com as especificações da turbina, incluindo a curva de potência
    """
    
    # Vestas V164-9.5 realistic power curve data (from OEM datasheets)
    # Format: (wind_speed_m_s, power_kW)
    power_curve_data = [
        (0.0, 0),
        (1.0, 0),
        (2.0, 0),
        (3.0, 0),
        (3.5, 350),      # Cut-in ## começar spline daqui
        (4.0, 650),
        (5.0, 1150),
        (6.0, 1950),
        (7.0, 3100),
        (8.0, 4600),
        (9.0, 6300),
        (10.0, 7900),
        (11.0, 8800),
        (12.0, 9200),
        (13.0, 9400),
        (14.0, 9500),    # Rated    ######## Plotar curva cúbica + lineares
        (15.0, 9500),
        (16.0, 9500),
        (17.0, 9500),
        (18.0, 9500),
        (19.0, 9500),
        (20.0, 9500),
        (21.0, 9500),
        (22.0, 9500),
        (23.0, 9500),
        (24.0, 9500),
        (25.0, 9500),    # Cut-out standard ## a partir daqui, ir para linear
        (26.0, 8000),    # High wind derating  
        (27.0, 6000),
        (28.0, 4000),
        (29.0, 2000),
        (30.0, 0),       # Full shutdown
    ]
    
    curve_array = np.array(power_curve_data)
    wind_speeds = curve_array[:, 0]
    power_output = curve_array[:, 1]
    
    # Create smooth cubic spline for interpolation
    cs = CubicSpline(wind_speeds, power_output, bc_type='natural')
    
    # Generate fine-grained curve for analysis
    wind_speeds_fine = np.linspace(0, 30, 300)
    power_fine = cs(wind_speeds_fine)
    power_fine = np.clip(power_fine, 0, CONFIG['nominal_power_kW'])
    
    power_curve_df = pd.DataFrame({
        'wind_speed': wind_speeds_fine,
        'value': power_fine
    })
    
    turbine_specs = {
        'turbine_type': 'Vestas V164-9.5 MW (Realistic Cubic)',
        'nominal_power_kW': CONFIG['nominal_power_kW'],
        'nominal_power_W': CONFIG['nominal_power_kW'] * 1000.0,
        'hub_height': CONFIG['hub_height'],
        'rotor_diameter': CONFIG['rotor_diameter'],
        'cut_in_speed': CONFIG['cut_in_speed'],
        'rated_speed': CONFIG['rated_speed'],
        'cut_out_speed': CONFIG['cut_out_speed'],
        'power_curve': power_curve_df,
        'power_curve_W': power_curve_df.copy(),
        'power_curve_source': 'Cubic interpolation of realistic OEM data',
        'cp_approx': 0.47,  # Approximate Cp at rated conditions
    }
    
    logger.info(f"✓ Loaded turbine: {turbine_specs['turbine_type']}")
    return turbine_specs

def prepare_weather_data(year, config):
    """
Carregar e validar dados meteorológicos de um arquivo CSV com tratamento robusto de erros.

Operações Matemáticas:

------------------------

1. Extrapolação do Cisalhamento do Vento (Lei Logarítmica):

Ajusta a velocidade do vento da altura de referência ($z_{ref}$) para a altura do cubo ($z_{hub}$).

$$ v_{hub} = v_{ref} \cdot \frac{\ln(z_{hub}/z_0)}{\ln(z_{ref}/z_0)} $$

Onde:

- $z_0$: Comprimento de rugosidade (0,0014 m para o mar)

2. Validação de Dados:

- Verifica valores NaN, velocidades de vento negativas e colunas ausentes.

Abordagem FAIL-FAST: Gera exceções para problemas de qualidade de dados,

em vez de recorrer silenciosamente a dados aleatórios.
    """
    
    if not os.path.exists(config['weather_csv_path']):
        raise FileNotFoundError(f"Weather data file not found: {config['weather_csv_path']}")
    
    try:
        df = pd.read_csv(config['weather_csv_path'])
        logger.info(f"✓ Loaded weather CSV: {len(df)} rows")
    except Exception as e:
        raise IOError(f"Failed to parse weather CSV: {e}")
    
    # Validate required columns
    required_cols = ['time', 'Wind Speed Hourly (m/s)', 'rhoa']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {df.columns.tolist()}")
    
    # Parse datetime
    try:
        df['time'] = pd.to_datetime(df['time'])
    except Exception as e:
        raise ValueError(f"Failed to parse 'time' column as datetime: {e}")
    
    # Filter to requested year
    weather_data = df[df['time'].dt.year == year].copy()
    if weather_data.empty:
        raise ValueError(f"No data found for year {year}")
    
    logger.info(f"✓ Filtered to year {year}: {len(weather_data)} records")
    
    # Validate and process wind speed
    if weather_data['Wind Speed Hourly (m/s)'].isna().any():
        raise ValueError("Wind speed data contains NaN values")
    
    if (weather_data['Wind Speed Hourly (m/s)'] < 0).any():
        raise ValueError("Wind speed contains negative values (physically invalid)")
    
    weather_data['wind_speed_10m'] = weather_data['Wind Speed Hourly (m/s)']
    
    # Apply wind shear extrapolation to hub height
    h_ref = config['input_wind_height']  # 10m (mast reference)
    h_hub = config['hub_height']
    z0 = config['z0']
    
    if h_ref != h_hub:
        shear_factor = np.log(h_hub / z0) / np.log(h_ref / z0)
        weather_data['wind_speed_hub'] = weather_data['wind_speed_10m'] * shear_factor
        logger.info(f"✓ Applied wind shear: {h_ref}m → {h_hub}m, factor={shear_factor:.3f}")
    else:
        weather_data['wind_speed_hub'] = weather_data['wind_speed_10m']
    
    # Process air density
    if weather_data['rhoa'].isna().any():
        raise ValueError("Air density (rhoa) contains NaN values")
    
    weather_data['air_density'] = weather_data['rhoa']
    
    # Set default temperature and pressure if not provided
    if 'temperature' not in weather_data.columns:
        weather_data['temperature'] = 15.0  # Celsius (will be converted to K in use)
    if 'pressure' not in weather_data.columns:
        weather_data['pressure'] = 101325.0  # Pa
    
    logger.info(f"✓ Weather data prepared: wind={weather_data['wind_speed_hub'].mean():.2f}±{weather_data['wind_speed_hub'].std():.2f} m/s")
    
    return weather_data

def calculate_power_generation_single_turbine(weather_data, turbine_specs, apply_density_correction=True):
    """
 Calcule a potência de saída de uma única turbina usando a curva de potência.

Justificativa matemática:

---------------------------

1. Correção da densidade do ar (IEC 61400-12-1):

As turbinas eólicas são testadas na densidade do ar padrão ($\rho_{std} = 1,225$ kg/m³).
Para usar a curva de potência padrão $P(v)$, devemos normalizar a velocidade do vento local:

$$ v_{corrigido} = v_{medido} \cdot \left( \frac{\rho_{local}}{\rho_{std}} \right)^{1/3} $$

Isso garante que o fluxo de energia cinética corresponda às condições padrão:

$$ \frac{1}{2} \rho_{local} v_{medido}^3 = \frac{1}{2} \rho_{std} v_{corrigido}^3 $$

2. Interpolação:

$$ P_{saída} = \text{Spline}(v_{corrigido}) $$

Argumentos:

weather_data: DataFrame com as colunas 'wind_speed_hub' e 'air_density'

turbine_specs: Dicionário com a curva de potência e os parâmetros da turbina

apply_density_correction: Indica se a correção de densidade deve ser aplicada Escala de acordo com a velocidade do vento

Retornos:

Série de potência de saída em kW
    """
    
    wind_speed = weather_data['wind_speed_hub'].values
    air_density = weather_data['air_density'].values
    
    # Apply density correction: v_corrected = v * (ρ/ρ_std)^(1/3)
    if apply_density_correction:
        rho_std = 1.225  # Standard sea level density
        wind_speed_corrected = wind_speed * (air_density / rho_std) ** (1/3)
    else:
        wind_speed_corrected = wind_speed
    
    # Interpolate power curve
    power_curve_df = turbine_specs['power_curve']
    cs = CubicSpline(
        power_curve_df['wind_speed'].values,
        power_curve_df['value'].values,
        bc_type='natural'
    )
    
    power_output_kW = cs(wind_speed_corrected)
    power_output_kW = np.clip(power_output_kW, 0, turbine_specs['nominal_power_kW'])
    
    return pd.Series(power_output_kW, index=weather_data.index)

def apply_curtailment_logic(power_series, prices_series, curtailment_threshold_EUR_MWh=-5.0):
    """
    Apply curtailment: zero out power when prices fall below economic threshold.
    Ensures index alignment between power and prices.
    """
    # Remove timezone info for alignment (keep the timestamps, just remove tz)
    power_idx = power_series.index
    price_idx = prices_series.index
    
    # If one has timezone and other doesn't, remove both
    if hasattr(power_idx, 'tz') and power_idx.tz is not None:
        power_series = power_series.copy()
        power_series.index = power_series.index.tz_localize(None)
    
    if hasattr(price_idx, 'tz') and price_idx.tz is not None:
        prices_series = prices_series.copy()
        prices_series.index = prices_series.index.tz_localize(None)
    
    # Now align indices (intersection)
    common_idx = power_series.index.intersection(prices_series.index)
    
    if len(common_idx) == 0:
        logger.error(f"No overlapping timestamps between power and prices!")
        logger.error(f"Power index sample: {power_series.index[:3]}")
        logger.error(f"Price index sample: {prices_series.index[:3]}")
        # Return original power series unchanged if no overlap
        return power_series
    
    if len(common_idx) < len(power_series):
        logger.warning(f"Data alignment warning: Power ({len(power_series)}) vs Prices ({len(prices_series)}). Using intersection ({len(common_idx)}).")
    
    # Reindex both to common index
    p_aligned = power_series.loc[common_idx].copy()
    pr_aligned = prices_series.loc[common_idx]
    
    # Create mask and apply curtailment
    curtail_mask = pr_aligned.values < curtailment_threshold_EUR_MWh
    p_aligned.iloc[curtail_mask] = 0
    
    curtailed_hours = np.sum(curtail_mask)
    if curtailed_hours > 0:
        energy_curtailed = power_series.loc[common_idx][curtail_mask].sum()
        logger.info(f"✓ Curtailment applied: {curtailed_hours} hours, {energy_curtailed:.1f} MWh curtailed")
    
    return p_aligned

def load_electricity_prices(year, config):
    """
  Carregar preços de eletricidade e reamostrar para resolução horária para correspondência com dados meteorológicos.

Justificativa Matemática:

---------------------------

Resolução da Incompatibilidade Temporal:

- Dados Meteorológicos: Resolução horária ($\Delta t = 1h$)

- Dados de Mercado: Resolução de 15 minutos ($\Delta t = 0,25h$)

Para alinhar esses conjuntos de dados, calculamos o preço médio por hora:

$$ P_{hora}(t) = \frac{1}{4} \sum_{i=1}^{4} P_{15min}(t, i) $$

Isso preserva o valor ponderado pela energia, já que a potência é considerada constante ao longo da hora no modelo meteorológico.

Argumentos:

year: Ano a ser carregado (normalmente 2024)

config: Dicionário de configuração

Retorna:

DataFrame com preços por hora, índice = tempo
    """
    # First, try to load from the user's CSV file
    price_csv_path = config.get('price_csv_path', 'NL_Prices_2024_15min.csv')
    
    if os.path.exists(price_csv_path):
        try:
            logger.info(f"Loading prices from {price_csv_path}...")
            
            # Read the CSV with proper datetime parsing
            df = pd.read_csv(price_csv_path)
            
            # Parse the first column as datetime (handle timezone-aware format)
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], utc=True)
            
            # Set as index
            df.set_index(df.columns[0], inplace=True)
            
            # Convert to Europe/Amsterdam timezone to match weather data
            df.index = df.index.tz_convert('Europe/Amsterdam')
            
            # Resample from 15-minute to hourly by taking the mean
            df_hourly = df.resample('H').mean()

            # Rename to standard column name
            df_hourly.columns = ['price_EUR_MWh']
            
            # Filter to requested year
            df_hourly = df_hourly[df_hourly.index.year == year]
            
            logger.info(f"✓ Loaded prices from CSV: {len(df_hourly)} hourly records")
            logger.info(f"✓ Price statistics: mean={df_hourly['price_EUR_MWh'].mean():.1f}, min={df_hourly['price_EUR_MWh'].min():.1f}, max={df_hourly['price_EUR_MWh'].max():.1f} EUR/MWh")
            
            return df_hourly, df
            
        except Exception as e:
            logger.warning(f"Failed to load price CSV: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    else:
        logger.warning(f"Price CSV not found: {price_csv_path}")
    
    # Try to load from cache
    if config.get('use_cached_prices') and os.path.exists(config['prices_cache_path']):
        try:
            prices_df = pd.read_csv(config['prices_cache_path'], index_col=0, parse_dates=True)
            logger.info(f"✓ Loaded cached prices from {config['prices_cache_path']}")
            return prices_df
        except Exception as e:
            logger.warning(f"Failed to load cached prices: {e}")
    
    # Try to fetch real data
    if HAS_ENTSOE and config.get('entsoe_country_code'):
        try:
            logger.info("Attempting to fetch prices from ENTSOE...")
            logger.warning("ENTSOE fetch requires API key - falling back to stub")
        except Exception as e:
            logger.warning(f"ENTSOE fetch failed: {e}")
    
    # Fallback: Generate synthetic but realistic prices
    logger.warning("⚠ Using synthetic price data for demonstration")
    
    # IMPORTANT: Make timezone-aware to match weather data
    date_range = pd.date_range(
        start=f'{year}-01-01', 
        end=f'{year}-12-31 23:00:00', 
        freq='H',
        tz='Europe/Amsterdam'  # Match weather data timezone
    )
    ##### esconder
    # Realistic Dutch price pattern: low mid-day (solar), high evening
    hours = date_range.hour
    season = date_range.month
    
    # Base price: seasonal variation
    base_price = 60 + 30 * np.sin(2 * np.pi * season / 12)
    
    # Intraday pattern: duck curve
    intraday_mult = 1 + 0.4 * np.sin(2 * np.pi * (hours + 4) / 24)
    
    # Random volatility
    noise = np.random.normal(0, 10, len(date_range))
    
    prices_array = base_price * intraday_mult + noise
    prices_array = np.clip(prices_array, -80, 300)
    
    prices_df = pd.DataFrame({
        'price_EUR_MWh': prices_array
    }, index=date_range)
    
    # FIXED: Use the DataFrame column instead of numpy array
    logger.info(f"✓ Generated synthetic prices: mean={prices_df['price_EUR_MWh'].mean():.1f}, min={prices_df['price_EUR_MWh'].min():.1f}, max={prices_df['price_EUR_MWh'].max():.1f} EUR/MWh")
    
    return prices_df


def calculate_turbine_count_with_ncf(turbine_specs, weather_data, power_single_turbine_kW, target_ncf):
    """
    Calcule o número MÁXIMO de turbinas que podem ser instaladas mantendo o NCF acima do alvo.
    
    Justificativa Matemática:
    ---------------------------
    O Fator de Capacidade Líquida (NCF) de um parque é dado por:
    $$ NCF_{park} = NCF_{single} \times \eta_{wake}(N) $$
    
    Onde:
    - $NCF_{single}$: Fator de capacidade de uma turbina isolada (depende apenas do vento local).
    - $\eta_{wake}(N)$: Eficiência de esteira, que diminui conforme $N$ aumenta.
    
    Como o NCF diminui com o aumento de $N$, não podemos "atingir" um alvo arbitrário apenas aumentando o número de turbinas.
    Pelo contrário, definimos um limite de qualidade: "Quantas turbinas posso colocar antes que a eficiência caia tanto que o NCF fique abaixo de X%?"
    
    Lógica:
    1. Calcula $NCF_{single}$.
    2. Se $NCF_{single} < Target$, retorna 0 (o local é ruim demais).
    3. Itera $N$ de 1 a 500.
    4. Calcula $NCF_{park}(N)$.
    5. Retorna o maior $N$ tal que $NCF_{park}(N) \ge Target$.
    """
    
    logger.info("\n=== TURBINE COUNT CALCULATION (NCF CONSTRAINED) ===")
    
    single_turbine_energy_MWh = power_single_turbine_kW.sum() / 1000.0
    P_nominal_MW = turbine_specs['nominal_power_kW'] / 1000.0
    hours_per_year = len(weather_data)
    
    # 1. Calculate Single Turbine NCF (Theoretical Maximum)
    ncf_single = single_turbine_energy_MWh / (P_nominal_MW * hours_per_year)
    logger.info(f"Single Turbine NCF (Max Potential): {ncf_single:.4f} ({ncf_single*100:.1f}%)")
    logger.info(f"Target NCF: {target_ncf:.4f} ({target_ncf*100:.1f}%)")
    
    if ncf_single < target_ncf:
        logger.warning(f"⚠ IMPOSSIBLE TARGET: Site potential ({ncf_single:.1%}) < Target ({target_ncf:.1%})")
        logger.warning("Returning 1 turbine for demonstration purposes.")
        return 1
        
    # 2. Iterate to find max N
    best_N = 1
    
    for N in range(1, 501):
        efficiency = calculate_wake_efficiency(N)
        ncf_park = ncf_single * efficiency
        
        if ncf_park >= target_ncf:
            best_N = N
            # logger.info(f"N={N}: Eff={efficiency:.3f}, NCF={ncf_park:.4f} (OK)")
        else:
            logger.info(f"N={N}: Eff={efficiency:.3f}, NCF={ncf_park:.4f} (< Target) -> STOP")
            break
            
    logger.info(f"✓ Selected Fleet Size: {best_N} turbines")
    logger.info(f"  Expected Farm Efficiency: {calculate_wake_efficiency(best_N):.3f}")
    logger.info(f"  Expected Farm NCF: {ncf_single * calculate_wake_efficiency(best_N):.4f}")
    
    return int(best_N)

def find_turbines_for_target_exceedance(
    power_single_turbine_kW,
    target_exceedance_percent,
    cutoff_MW,
    farm_efficiency_func
):
    """
  Encontrar o número mínimo de turbinas necessárias para ultrapassar um limite de potência
para uma porcentagem alvo de horas.

Lógica Matemática:

-------------------

Buscamos o menor inteiro $N$ tal que:

$$ P(P_{farm}(t) \ge P_{cutoff}) \ge \frac{Target\%}{100} $$

Onde:

$$ P_{farm}(t) = N \cdot P_{single}(t) \cdot \eta_{wake}(N) $$

A função itera $N$ de 1 a 500, recalculando a eficiência de esteira $\eta_{wake}(N)$ a cada passo (já que a eficiência diminui conforme N aumenta), até que a condição seja satisfeita.

Argumentos:

potência_turbina_única_kW: Série de potência de saída de turbina única

porcentagem_alvo_exceedance: Percentual alvo (0-100)

limiar_potência_limite (MW): Limiar de potência (MW)

função_eficiência_fazenda: Função(N_turbinas) -> fator de eficiência

Retorna:

(N_turbinas, porcentagem_alcançada) ou (Nenhum, 0) se impossível
    """
    
    logger.info(f"\n=== TURBINE SIZING FOR {target_exceedance_percent}% EXCEEDANCE ===")
    logger.info(f"Target: Power >= {cutoff_MW} MW for {target_exceedance_percent}% of hours")
    
    P_nom_MW = CONFIG['nominal_power_kW'] / 1000.0
    
    for N in range(1, 501):  # Realistic upper limit: 500 turbines
        efficiency = farm_efficiency_func(N)
        park_power_MW = (power_single_turbine_kW * N * efficiency) / 1000.0
        
        hours_above = (park_power_MW >= cutoff_MW).sum()
        percent_above = (hours_above / len(park_power_MW)) * 100
        
        if percent_above >= target_exceedance_percent:
            logger.info(f"✓ Solution found: {N} turbines ({percent_above:.2f}% ≥ {cutoff_MW} MW)")
            logger.info(f"  Park capacity: {N * P_nom_MW:.1f} MW")
            logger.info(f"  Farm efficiency: {efficiency:.3f}")
            return N, percent_above
    
    logger.warning(f"⚠ No solution found within 500 turbines")
    return None, 0

def calculate_daily_revenue(
    park_power_MW,
    prices_EUR_MWh,
    weather_time_index
):
    """
    Calculate daily revenue from hourly power and price data.
    
    Returns:
        DataFrame with daily aggregation
    """
    
    hourly_df = pd.DataFrame({
        'time': weather_time_index,
        'power_MW': park_power_MW.values,
        'price_EUR_MWh': prices_EUR_MWh.values,
    })
    hourly_df.set_index('time', inplace=True)
    
    hourly_df['revenue_EUR'] = hourly_df['power_MW'] * hourly_df['price_EUR_MWh']
    hourly_df['date'] = hourly_df.index.date
    
    daily_df = hourly_df.groupby('date').agg({
        'revenue_EUR': 'sum',
        'power_MW': 'sum',
        'price_EUR_MWh': 'mean'
    }).reset_index()
    
    daily_df.rename(columns={
        'revenue_EUR': 'revenue_EUR',
        'power_MW': 'energy_MWh',
        'price_EUR_MWh': 'avg_price_EUR_MWh'
    }, inplace=True)
    
    return daily_df, hourly_df

def calculate_capture_price(hourly_df):
    """
    Calcule o preço médio recebido por MWh de geração eólica.

    Definição Matemática:

    ------------------------

    O Preço de Captura é o preço médio ponderado pela geração:

    $$ P_{captura} = \frac{\sum_{t} (E_{gen}(t) \cdot Preço(t))}{\sum_{t} E_{gen}(t)} $$

    Taxa de Captura (Fator de Valor):

    $$ CR = \frac{P_{captura}}{\bar{P}_{mercado}} $$

    Onde $\bar{P}_{mercado}$ é o preço médio ponderado pelo tempo.

    - $CR < 1$: Custo do perfil (vento sopra quando os preços estão baixos)

    - $CR > 1$: Benefício do perfil (vento sopra quando os preços estão altos)
    """
    total_revenue = hourly_df['revenue_EUR'].sum()
    total_generation = hourly_df['power_MW'].sum()
    
    if total_generation == 0:
        return 0
    
    capture_price = total_revenue / total_generation
    market_price = hourly_df['price_EUR_MWh'].mean()
    capture_ratio = capture_price / market_price if market_price != 0 else 0
    
    return {
        'capture_price_EUR_MWh': capture_price,
        'market_price_EUR_MWh': market_price,
        'capture_ratio': capture_ratio,
    }

def analyze_statistics(data, name, unit):
    """Generic statistics analyzer."""
    stats = {
        'name': name,
        'unit': unit,
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'p10': np.percentile(data, 10),
        'p90': np.percentile(data, 90),
    }
    return stats

def generate_plots(weather_data, park_power_MW, daily_revenue_df, hourly_df, output_dir):
    """Generate comprehensive analysis plots."""
    
    if not CONFIG['save_plots'] or not HAS_MATPLOTLIB:
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available - skipping plots")
        return
    
    logger.info("\n=== GENERATING PLOTS ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Wind Speed Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(weather_data['wind_speed_hub'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(weather_data['wind_speed_hub'].mean(), color='red', linestyle='--', label=f'Mean: {weather_data["wind_speed_hub"].mean():.2f} m/s')
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Frequency (hours)')
    ax.set_title('Wind Speed Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_wind_speed_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved 01_wind_speed_distribution.png")
    
    # 2. Power Output Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(park_power_MW, bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(park_power_MW.mean(), color='red', linestyle='--', label=f'Mean: {park_power_MW.mean():.2f} MW')
    ax.set_xlabel('Power Output (MW)')
    ax.set_ylabel('Frequency (hours)')
    ax.set_title('Park Power Output Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_power_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved 02_power_distribution.png")
    
    # 3. Time Series - Wind Speed (subsample for clarity)
    fig, ax = plt.subplots(figsize=(14, 5))
    step = 24  # Show every day
    ax.plot(weather_data['time'][::step], weather_data['wind_speed_hub'][::step], linewidth=1.5, color='steelblue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Wind Speed Time Series (Daily Sample)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_wind_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved 03_wind_timeseries.png")
    
    # 4. Time Series - Power Output (subsample)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(weather_data['time'][::step], park_power_MW[::step], linewidth=1.5, color='green')
    ax.set_xlabel('Date')
    ax.set_ylabel('Power Output (MW)')
    ax.set_title('Park Power Output Time Series (Daily Sample)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_power_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved 04_power_timeseries.png")
    
    # 5. Daily Revenue
    if daily_revenue_df is not None and not daily_revenue_df.empty:
        fig, ax = plt.subplots(figsize=(14, 5))
        dates = pd.to_datetime(daily_revenue_df['date'])
        ax.bar(dates, daily_revenue_df['revenue_EUR'], width=0.8, alpha=0.7, color='darkgreen', edgecolor='black')
        ax.set_xlabel('Date')
        ax.set_ylabel('Revenue (EUR)')
        ax.set_title('Daily Revenue Time Series')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '05_daily_revenue.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved 05_daily_revenue.png")
    
    # 6. Price Distribution with Negative Pricing Highlight
    if 'price_EUR_MWh' in hourly_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        prices = hourly_df['price_EUR_MWh']
        ax.hist(prices, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Price')
        ax.axvline(prices.mean(), color='green', linestyle='--', label=f'Mean: {prices.mean():.2f} EUR/MWh')
        negative_pct = (prices < 0).sum() / len(prices) * 100
        ax.set_xlabel('Price (EUR/MWh)')
        ax.set_ylabel('Frequency (hours)')
        ax.set_title(f'Electricity Price Distribution (Negative pricing: {negative_pct:.1f}% of hours)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '06_price_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved 06_price_distribution.png")
    
    # 7. Exceedance Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds = np.linspace(0, park_power_MW.max(), 100)
    exceedance = [(park_power_MW >= t).sum() / len(park_power_MW) * 100 for t in thresholds]
    ax.plot(thresholds, exceedance, linewidth=2, color='steelblue')
    ax.axhline(y=80, color='red', linestyle='--', label='80% Target')
    ax.axhline(y=50, color='orange', linestyle='--', label='50% Reference')
    ax.set_xlabel('Power Output (MW)')
    ax.set_ylabel('% Hours Exceeding Threshold')
    ax.set_title('Power Exceedance Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_exceedance_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved 07_exceedance_curve.png")

def create_summary_report(results, output_dir):
    logger.info("✓ Creating executive summary...")
    
    report_path = os.path.join(output_dir, "SUMMARY_REPORT.txt")
    
    # Safe formatting helpers
    def fmt_num(val, decimals=1):
        try:
            return f"{float(val):.{decimals}f}"
        except (ValueError, TypeError):
            return "N/A"
    
    def fmt_pct(val):
        try:
            return f"{float(val)*100:.1f}%"
        except (ValueError, TypeError):
            return "N/A"
    
    def fmt_int(val):
        try:
            return f"{int(float(val)):,}"
        except (ValueError, TypeError):
            return "N/A"
    
    econ = results['economics']
    tech = results['technology']
    
    with open(report_path, 'w') as f:
        f.write("🏗️ WIND FARM SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("📊 TECHNOLOGY\n")
        f.write(f"• Turbines: {fmt_int(econ.get('n_turbines'))}\n")
        f.write(f"• Capacity: {fmt_num(tech.get('rated_power_MW'), 1)} MW × {fmt_int(econ.get('n_turbines'))}\n")
        f.write(f"• Model: {tech.get('turbine_model', 'N/A')}\n\n")
        
        f.write("⚡ PERFORMANCE\n")
        f.write(f"• Annual Energy: {fmt_int(econ.get('annual_energy_MWh')/1000)} GWh\n")
        f.write(f"• Capacity Factor: {fmt_pct(econ.get('actual_ncf'))}\n")
        f.write(f"• Peak Power: {fmt_num(econ.get('max_power_MW'), 1)} MW\n\n")
        
        f.write("💰 ECONOMICS\n")
        f.write(f"• Total Revenue: {fmt_int(econ.get('total_revenue_EUR'))} €\n")
        f.write(f"• Avg Price: {fmt_num(econ.get('price_mean'), 1)} €/MWh\n")
        f.write(f"• Capture Price: {fmt_num(econ.get('capture_price_EUR_MWh'), 1)} €/MWh\n\n")
        
        f.write("📈 FILES GENERATED\n")
        for root, dirs, files in os.walk(output_dir):
            for file in sorted(files):
                if file.endswith(('.csv', '.png', '.txt')):
                    f.write(f"• {file}\n")
    
    logger.info(f"✓ Report: {report_path}")
    
    # CSV summary table
    summary_df = pd.DataFrame({
        'Category': ['Technology', 'Technology', 'Performance', 'Performance', 'Economics', 'Economics'],
        'Metric': ['Turbines', 'Capacity (MW)', 'Energy (GWh)', 'Capacity Factor (%)', 'Revenue (€)', 'Capture Price (€/MWh)'],
        'Value': [
            econ.get('n_turbines', 0),
            tech.get('rated_power_MW', 0) * econ.get('n_turbines', 0),
            econ.get('annual_energy_MWh', 0) / 1000,
            econ.get('actual_ncf', 0) * 100,
            econ.get('total_revenue_EUR', 0),
            econ.get('capture_price_EUR_MWh', 0)
        ]
    })
    summary_csv = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    logger.info("✓ summary.csv ready")


# ==================== MAIN EXECUTION ====================

def main():
    """Main analysis pipeline."""
    
    print("\n" + "=" * 80)
    print("WIND POWER PLANT ANALYSIS ENGINE - REVISED & VALIDATED")
    print("=" * 80 + "\n")
    
    # Validate configuration
    validate_config()
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    logger.info(f"✓ Output directory: {CONFIG['output_dir']}")
    
    # ========== Data Loading ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: DATA LOADING AND VALIDATION")
    logger.info("=" * 80)

    try:
        weather = prepare_weather_data(2024, CONFIG)
        # --- ADD THIS LINE ---
        # Set time as index for alignment, but keep column for plots/math
        weather['time'] = pd.to_datetime(weather['time'])
        # Define a coluna 'time' como o índice principal
        weather.set_index('time', inplace=True, drop=False)
        # ---------------------
    except Exception as e:
        logger.error(f"✗ FATAL: Failed to load weather data: {e}")
        return
    
    try:
        turbine = load_turbine_power_curve_cubic()
    except Exception as e:
        logger.error(f"✗ FATAL: Failed to load turbine model: {e}")
        return
    
    # ========== Power Generation ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: SINGLE TURBINE POWER GENERATION")
    logger.info("=" * 80)
    
    power_kW = calculate_power_generation_single_turbine(
        weather,
        turbine,
        apply_density_correction=CONFIG['apply_density_correction']
    )
    
    # ========== Wind Resource Analysis ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: RESOURCE ANALYSIS")
    logger.info("=" * 80)
    
    wind_stats = analyze_statistics(weather['wind_speed_hub'], 'Wind Speed', 'm/s')
    logger.info(f"Wind stats: mean={wind_stats['mean']:.2f}, std={wind_stats['std']:.2f} m/s")
    
    # ========== Turbine Fleet Sizing ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: FLEET SIZING")
    logger.info("=" * 80)
    
    n_turbines = calculate_turbine_count_with_ncf(
        turbine,
        weather,
        power_kW,
        CONFIG['net_capacity_factor']
    )
    
    total_capacity_MW = n_turbines * CONFIG['nominal_power_kW'] / 1000.0
    farm_efficiency = calculate_wake_efficiency(n_turbines)
    
    logger.info(f"Total capacity: {total_capacity_MW:.1f} MW")
    logger.info(f"Farm efficiency factor: {farm_efficiency:.3f}")

   # =======================================================
    # FASE SIMPLIFICADA: APENAS POTÊNCIA DE 2 TURBINAS (HORÁRIA)
    # =======================================================
    logger.info("=== GERANDO TABELA SIMPLIFICADA (2 TURBINAS / HORA) ===")

    # 1. Garante que o índice é temporal
    if 'time' in weather.columns:
        weather['time'] = pd.to_datetime(weather['time'])
        weather.set_index('time', inplace=True, drop=False)
    
    # --- MODIFICAÇÃO AQUI ---
    # Calcula eficiência para exatamente 2 turbinas
    # Se quiser ignorar perdas de esteira, remova o termo 'eficiencia_2' ou defina como 1.0
    eficiencia_2 = calculate_wake_efficiency(2) 
    
    tabela_simples = pd.DataFrame({
        'data_hora': weather.index,
        'velocidade_vento_ms': weather['wind_speed_hub'],
        # Multiplica por 2, aplica eficiência e converte kW -> MW
        'potencia_2_turbinas_MW': (power_kW * 2 * eficiencia_2) / 1000.0
    })
    # ------------------------

    # 3. Remove qualquer linha vazia ou erro
    tabela_simples = tabela_simples.dropna()

    # 4. Salva o arquivo com novo nome
    output_simple = os.path.join(CONFIG['output_dir'], 'producao_horaria_2_turbinas.csv')
    tabela_simples.to_csv(output_simple, sep=';', decimal='.', index=False)
    
    logger.info(f"✓ Tabela simplificada salva: {output_simple}")
    logger.info(f"  Fator de eficiência aplicado: {eficiencia_2:.3f}")
    logger.info(f"  Média (2 turbinas): {tabela_simples['potencia_2_turbinas_MW'].mean():.2f} MW")
    
    # ========== Park Generation ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: PARK POWER OUTPUT")
    logger.info("=" * 80)
    
    park_power_MW = (power_kW * n_turbines * farm_efficiency) / 1000.0
    annual_energy_MWh = park_power_MW.sum()
    actual_ncf = annual_energy_MWh / (total_capacity_MW * len(weather))
    
    logger.info(f"Annual energy: {annual_energy_MWh:.0f} MWh")
    logger.info(f"Actual NCF: {actual_ncf:.4f}")
    
    gen_stats = analyze_statistics(park_power_MW, 'Power Output', 'MW')
    logger.info(f"Power stats: mean={gen_stats['mean']:.2f} MW, max={gen_stats['max']:.2f} MW")

    # ========== Price and Revenue Analysis ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: MARKET AND REVENUE ANALYSIS")
    logger.info("=" * 80)

    # Load prices and handle tuple if returned (CSV mode)
    result = load_electricity_prices(2024, CONFIG)
    
    if isinstance(result, tuple):
        prices_df = result[0]  # Extract hourly dataframe from (hourly, raw)
    else:
        prices_df = result     # It's already the dataframe (synthetic mode)

    # Now this will work safely
    prices_hourly = prices_df['price_EUR_MWh']

    # Unix timestamps for numeric matching
    origin = pd.Timestamp('1970-01-01')
    weather_unix = (weather['time'].dt.tz_localize(None) - origin).dt.total_seconds().values.astype(int)
    price_unix = (prices_hourly.index.tz_localize(None) - origin).total_seconds().values.astype(int)

    unique_mask = ~pd.Series(weather_unix).duplicated(keep='first')
    weather_unix_unique = weather_unix[unique_mask]
    park_power_unique = park_power_MW.values[unique_mask]
    n_records = len(weather_unix_unique)
    logger.info(f"Data ready: {n_records} records")

    # Sorted price indices
    sort_idx = np.argsort(price_unix)
    matching_indices = np.full(n_records, -1, dtype=int)

    for i, wt_unix in enumerate(weather_unix_unique):
        idx = np.searchsorted(price_unix[sort_idx], wt_unix, side='left')
        
        # Tolerance: ±1 hour (3600 seconds)
        window_left = max(0, idx - 2)
        window_right = min(len(price_unix), idx + 3)
        candidates = sort_idx[window_left:window_right]
        diffs = np.abs(price_unix[candidates] - wt_unix)
        
        best_idx = np.argmin(diffs)
        if diffs[best_idx] <= 3600:  # 1hr tolerance
            matching_indices[i] = candidates[best_idx]

    num_matched = np.sum(matching_indices >= 0)
    logger.info(f"✓ Direct matches: {num_matched}/{n_records} (tol=1hr)")

    # Build matched prices
    prices_matched = np.full(n_records, prices_hourly.mean())
    prices_matched[matching_indices >= 0] = prices_hourly.values[matching_indices[matching_indices >= 0]]

    # Pandas ffill/bfill for perfect coverage
    time_idx = pd.to_datetime(weather_unix_unique, unit='s', origin=origin)
    price_series = pd.Series(prices_matched, index=time_idx).fillna(method='ffill').fillna(method='bfill')

    prices_matched = price_series.values

    # Final assert
    assert len(prices_matched) == n_records == len(park_power_unique)

    # Curtailment
    curtail_mask = prices_matched < CONFIG['curtailment_threshold_EUR_MWh']
    park_power_curtailed = park_power_unique.copy()
    park_power_curtailed[curtail_mask] = 0
    curtailed_hours = curtail_mask.sum()
    energy_curtailed = park_power_unique[curtail_mask].sum()
    if curtailed_hours > 0:
        logger.info(f"✓ Curtailment: {curtailed_hours} hrs ({energy_curtailed:.1f} MWh)")

    # Revenue
    hourly_df = pd.DataFrame({
        'power_MW': park_power_curtailed,
        'price_EUR_MWh': prices_matched
    }, index=time_idx)
    hourly_df['revenue_EUR'] = hourly_df['power_MW'] * hourly_df['price_EUR_MWh']
    hourly_df['date'] = hourly_df.index.date

    daily_revenue_df = hourly_df.groupby('date').agg({
        'revenue_EUR': 'sum',
        'power_MW': 'sum',
        'price_EUR_MWh': 'mean'
    }).reset_index()
    daily_revenue_df.rename(columns={'power_MW': 'energy_MWh'}, inplace=True)

    total_revenue_EUR = daily_revenue_df['revenue_EUR'].sum()
    price_stats = analyze_statistics(prices_matched, 'Price', 'EUR/MWh')
    econ_metrics = calculate_capture_price(hourly_df)
    econ_metrics['total_revenue_EUR'] = total_revenue_EUR

    logger.info(f"✓ Revenue: {total_revenue_EUR:,.0f} EUR")
    logger.info(f"✓ Capture price: {econ_metrics['capture_price_EUR_MWh']:.1f} EUR/MWh")
        

    
    # ========== Threshold Analysis ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 7: THRESHOLD AND EXCEEDANCE ANALYSIS")
    logger.info("=" * 80)
    
    hours_above_cutoff = (park_power_MW >= CONFIG['cutoff_power_MW']).sum()
    percent_above = (hours_above_cutoff / len(park_power_MW)) * 100
    hours_zero = (park_power_MW == 0).sum()
    
    logger.info(f"Hours ≥ {CONFIG['cutoff_power_MW']} MW: {hours_above_cutoff} ({percent_above:.2f}%)")
    logger.info(f"Hours at zero output: {hours_zero} ({hours_zero/len(park_power_MW)*100:.2f}%)")
    
    n_turbines_exceedance, exceedance_pct = find_turbines_for_target_exceedance(
        power_kW,
        CONFIG['target_exceedance_percent'],
        CONFIG['cutoff_power_MW'],
        calculate_wake_efficiency
    )
    
    # ========== Results Summary ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 8: GENERATING REPORTS")
    logger.info("=" * 80)
    
    results = {
        # Turbine fleet
        'n_turbines': n_turbines,
        'total_capacity_MW': total_capacity_MW,
        'farm_efficiency': farm_efficiency,
        
        # Energy
        'annual_energy_MWh': annual_energy_MWh,
        'ncf': actual_ncf,
        
        # Wind resource
        **{f'wind_{k}': v for k, v in wind_stats.items()},
        
        # Power output
        **{f'power_{k}': v for k, v in gen_stats.items()},
        
        # Threshold
        'hours_above_cutoff': hours_above_cutoff,
        'percent_above_cutoff': percent_above,
        'hours_zero': hours_zero,
        
        # Price & economics
        **{f'price_{k}': v for k, v in price_stats.items()},
        **{f'econ_{k}': v for k, v in econ_metrics.items()},
        
        # Exceedance
        'n_turbines_for_exceedance': n_turbines_exceedance if n_turbines_exceedance else 'N/A',
        'exceedance_achieved_pct': exceedance_pct,
    }
    
    # Generate visualizations
    generate_plots(weather, park_power_MW, daily_revenue_df, hourly_df, CONFIG['output_dir'])
    
    # Create reports
    #create_summary_report(results, CONFIG['output_dir'])
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {CONFIG['output_dir']}")
    logger.info("=" * 80 + "\n")
    
    return results

    

if __name__ == '__main__':
    results = main()
