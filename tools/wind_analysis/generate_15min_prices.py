#!/usr/bin/env python3
"""
Generate 15-minute resolution price data from existing hourly data.

Since ENTSO-E API is temporarily unavailable, this script creates realistic
15-minute price data by interpolating the existing hourly prices_2024.csv
and adding realistic intraday variation.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("GERAÇÃO DE PREÇOS 15 MINUTOS A PARTIR DE DADOS HORÁRIOS")
print("="*70)

# 1. Carregar dados horários existentes
data_dir = Path(__file__).parent.parent / 'h2_plant' / 'data'
hourly_file = data_dir / 'prices_2024.csv'

print(f"\nCarregando: {hourly_file}")
hourly_prices = np.loadtxt(hourly_file, delimiter=',')
print(f"  ✓ {len(hourly_prices)} preços hourly carregados")
print(f"  • Min: {hourly_prices.min():.2f} EUR/MWh")
print(f"  • Max: {hourly_prices.max():.2f} EUR/MWh")
print(f"  • Média: {hourly_prices.mean():.2f} EUR/MWh")

# 2. Criar timestamps
start_date = pd.Timestamp('2024-01-01', tz='Europe/Amsterdam')
hourly_index = pd.date_range(start=start_date, periods=len(hourly_prices), freq='h')

# Create DataFrame
df_hourly = pd.Series(hourly_prices, index=hourly_index)

print(f"\n" + "-"*70)
print("CRIANDO DADOS 15 MINUTOS...")
print("-"*70)

# 3. Resample para 15 minutos com interpolação linear
df_15min = df_hourly.resample('15min').asfreq()
df_15min = df_15min.interpolate(method='linear')

# 4. Adicionar variação intraday realista
# Preços reais variam ~2-5% dentro de cada hora devido a:
# - Variações de demanda
# - Flutuações de oferta
# - Ajustes de mercado
np.random.seed(42)  # Para reprodutibilidade
variation = np.random.normal(0, 0.015, len(df_15min))  # ~1.5% std dev
df_15min = df_15min * (1 + variation)

# Garantir que não ficamos negativos
df_15min = df_15min.clip(lower=0.0)

# 5. Preencher NaN se houver
df_15min = df_15min.fillna(method='bfill').fillna(method='ffill')

print(f"  ✓ Dados 15 minutos criados")
print(f"  • Total de pontos: {len(df_15min)}")
print(f"  • Esperado (1 ano @ 15min): ~35,040")
print(f"  • Min: {df_15min.min():.2f} EUR/MWh")
print(f"  • Max: {df_15min.max():.2f} EUR/MWh")
print(f"  • Média: {df_15min.mean():.2f} EUR/MWh")

# 6. Salvar arquivos
print(f"\n" + "-"*70)
print("SALVANDO ARQUIVOS...")
print("-"*70)

# Arquivo com timestamp
output_dir = Path(__file__).parent
filename_full = output_dir / 'NL_Prices_2024_15min.csv'
df_15min.to_csv(filename_full, header=['price_eur_mwh'])
print(f"✓ Arquivo completo: {filename_full}")

# Arquivo simples (sem header)
filename_simple = output_dir / 'prices_2024.csv'
np.savetxt(filename_simple, df_15min.values, delimiter=',', fmt='%.15e')
print(f"✓ Arquivo simplificado: {filename_simple}")

# 7. Copiar para h2_plant/data com backup
dest_file = data_dir / 'prices_2024.csv'
backup_file = data_dir / 'prices_2024_hourly_backup.csv'

import shutil
# Criar backup do arquivo hourly original
shutil.copy(dest_file, backup_file)
print(f"  ⓘ Backup hourly: {backup_file}")

# Copiar novo arquivo 15min
shutil.copy(filename_simple, dest_file)
print(f"✓ Novo arquivo copiado para: {dest_file}")

# 8. Mostrar estatísticas de comparação
print(f"\n" + "="*70)
print("COMPARAÇÃO HOURLY vs 15MIN")
print("="*70)
print(f"{'Métrica':<20} {'Hourly':<15} {'15-min':<15} {'Diferença':<15}")
print("-"*70)
print(f"{'Pontos de dados':<20} {len(hourly_prices):<15} {len(df_15min):<15} {len(df_15min) - len(hourly_prices):<15}")
print(f"{'Min (EUR/MWh)':<20} {hourly_prices.min():<15.2f} {df_15min.min():<15.2f} {df_15min.min() - hourly_prices.min():<15.2f}")
print(f"{'Max (EUR/MWh)':<20} {hourly_prices.max():<15.2f} {df_15min.max():<15.2f} {df_15min.max() - hourly_prices.max():<15.2f}")
print(f"{'Média (EUR/MWh)':<20} {hourly_prices.mean():<15.2f} {df_15min.mean():<15.2f} {df_15min.mean() - hourly_prices.mean():<15.2f}")
print(f"{'Desvio Padrão':<20} {hourly_prices.std():<15.2f} {df_15min.std():<15.2f} {df_15min.std() - hourly_prices.std():<15.2f}")

# Mostrar amostra
print(f"\n" + "="*70)
print("PRIMEIROS 20 VALORES (15 MIN)")
print("="*70)
print(df_15min.head(20))

print(f"\n" + "="*70)
print("CONVERSÃO CONCLUÍDA COM SUCESSO!")
print("="*70)
print("\nNOTA: Para dados REAIS da ENTSO-E, execute Script_ENTSO-E.md quando")
print("      a API estiver disponível. Este arquivo usa interpolação + variação")
print("      estocástica para aproximar preços intraday de 15 minutos.")
