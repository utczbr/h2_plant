import matplotlib.pyplot as plt
import numpy as np
import os

def gerar_graficos(df):
    """
    Gera os gráficos baseados nos resultados da simulação e salva no caminho específico.
    Versão atualizada para suportar múltiplos compressores e balanço de reciclo.
    """
    # Definição do caminho de saída conforme seu padrão
    output_dir = r'C:\Users\tusaw\OneDrive\Documentos\projeto hidrogenio\ATR\pos atr\gráficos'
    
    # Cria o diretório caso ele não exista
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Eixo X comum (Fluxo de O2)
    x = df['F_O2'] 
    
    # --- 1. Gráfico de Erros (H08 e H09) ---
    erro_h08 = np.abs(df['Q_H08'] - df['Q_H08_plan'])
    erro_h09 = np.abs(df['Q_H09'] - df['Q_H09_plan'])

    plt.figure(figsize=(10, 6))
    plt.plot(x, erro_h08, label='Erro Absoluto H08', color='blue', marker='o')
    plt.plot(x, erro_h09, label='Erro Absoluto H09', color='red', marker='s')
    plt.title('Diferença de Carga Térmica: Modelo vs Planejado')
    plt.xlabel('Fluxo de O2')
    plt.ylabel('Erro (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, '01_erro_validacao.png'))
    plt.close()

    # --- 2. Gráfico de Água Retirada (Ciclones e Equipamentos) ---
    plt.figure(figsize=(10, 6))
    # Note a correção do nome da coluna para H2O_rem_Cic1
    plt.plot(x, df['H2O_rem_Cic1'], label='1º Ciclone (Pós-HEX)', linestyle='-')
    plt.plot(x, df['H2O_rem_Cic2'], label='2º Ciclone (Pós-Chiller)', linestyle='--', color='orange')
    plt.plot(x, df['H2O_rem_Coal'], label='Coalescedor', linestyle='-.')
    plt.plot(x, df['H2O_rem_PSA'], label='PSA (Secagem Final)', linestyle=':')
    plt.title('Massa de Água Retirada por Equipamento')
    plt.xlabel('Fluxo de O2')
    plt.ylabel('Água Removida (kg/h)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, '02_agua_retirada_processo.png'))
    plt.close()

    # --- 3. Gráfico de Potência Consumida (Stackplot) ---
    plt.figure(figsize=(10, 6))
    plt.stackplot(x, 
                  df['W_Chil'], 
                  df['W_Comp1'], 
                  df['W_Comp2'], 
                  labels=['Chiller', 'Compressor 1', 'Compressor 2'],
                  alpha=0.7)
    plt.title('Distribuição do Consumo de Potência Total')
    plt.xlabel('Fluxo de O2')
    plt.ylabel('Potência (kW)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(output_dir, '03_consumo_potencia.png'))
    plt.close()

    # --- 4. Gráfico de Balanço de Massa ATR (Reciclo vs Reposição) ---
    plt.figure(figsize=(10, 6))
    plt.bar(x, df['F_H2O_Recup'], label='Água Reciclada (Ciclones)', color='blue', alpha=0.6)
    plt.bar(x, df['F_H2O_Makeup'], bottom=df['F_H2O_Recup'], label='Água Reposição (Makeup)', color='red', alpha=0.6)
    plt.title('Composição da Água Alimentada ao ATR (Balanço Final)')
    plt.xlabel('Fluxo de O2')
    plt.ylabel('Vazão Total (kg/h)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, '04_balanco_reciclo_atr.png'))
    plt.close()

    # --- 5. Gráfico de Temperaturas de Saída ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, df['T_H2O_Recup'], label='T Mistura Ciclones', color='green')
    plt.plot(x, df['T_H2O_ATR_Final'], label='T Final (Reciclo + Makeup)', color='darkblue', linewidth=2)
    plt.title('Evolução da Temperatura da Água Recuperada')
    plt.xlabel('Fluxo de O2')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, '05_temperaturas_mistura.png'))
    plt.close()

    print(f"Os 5 gráficos foram salvos com sucesso em: {output_dir}")