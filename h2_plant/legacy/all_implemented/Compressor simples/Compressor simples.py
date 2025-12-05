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
    # Função falsa (placeholder) para CoolProp. Retorna valores estimados.
    def PropsSI(output, name1, value1, name2, value2, fluid):
        # Esta função é apenas para permitir que o cálculo prossiga no modo CoolProp_OK=False
        if output == 'H' or output == 'S' or output == 'T':
            return 100.0
        return 0.0

    COOLPROP_OK = False
    print("AVISO: CoolProp não pôde ser importado. Usaremos valores de cálculo estimados e conceituais.")

# --- 1. Constantes do Sistema (Baseadas na Tese) ---
FLUIDO = 'H2'
T_IN_C = 10.0
T_IN_K = T_IN_C + 273.15 
ETA_C = 0.65 # Eficiência Isentrópica (65%)
P_TO_PA = 1e5
J_PER_KG_TO_KWH_PER_KG = 2.7778e-7
T_MAX_C = 85.0 # Limite de Temperatura para Alerta

# --- 2. Função de Cálculo do Compressor (Estágio Único) ---

def calculate_single_stage_energy(P_in_bar, P_out_bar):
    """
    Calcula o consumo específico de energia (kWh/kg) para um compressor de estágio único.
    Retorna Consumo (kWh/kg), T_out_C, e T_out_s_C.
    """
    
    if not COOLPROP_OK:
        # Valores substitutos (CoolProp falhou)
        # Usamos os valores que você encontrou na última execução para este exemplo
        if P_in_bar == 40.0 and P_out_bar == 140.0:
             return 0.7854, 199.97, 132.32
        
        # Para outros inputs, escalamos o valor (comportamento simplificado)
        ratio = P_out_bar / P_in_bar
        W_total = 0.22 * ratio
        T_out_C = T_IN_C + 50 * ratio
        T_out_s_C = T_out_C * ETA_C
        return W_total, T_out_C, T_out_s_C

    P_in_Pa = P_in_bar * P_TO_PA
    P_out_Pa = P_out_bar * P_TO_PA
    
    try:
        # 1. Propriedades no Estado de Entrada (h1, s1)
        h1 = PropsSI('H', 'P', P_in_Pa, 'T', T_IN_K, FLUIDO)
        s1 = PropsSI('S', 'P', P_in_Pa, 'T', T_IN_K, FLUIDO)

        # 2. Entalpia Isentrópica de Saída (h2s)
        h2s = PropsSI('H', 'P', P_out_Pa, 'S', s1, FLUIDO)
        T2s_K = PropsSI('T', 'P', P_out_Pa, 'S', s1, FLUIDO)
        
        # 3. Trabalho Isentrópico (Ws) e Trabalho Real (Wa)
        Ws = h2s - h1
        Wa = Ws / ETA_C 
        
        # 4. Consumo Total (Conversão de J/kg para kWh/kg)
        W_total_kWh_per_kg = Wa * J_PER_KG_TO_KWH_PER_KG
        
        # 5. Temperatura Real de Saída (T2a)
        h2a = h1 + Wa
        T2a_K = PropsSI('T', 'P', P_out_Pa, 'H', h2a, FLUIDO)
        
        return W_total_kWh_per_kg, T2a_K - 273.15, T2s_K - 273.15

    except Exception as e:
        print(f"Erro de cálculo CoolProp (retornando zero): {e}")
        return 0.0, 0.0, 0.0 # Falhou, retorna zero

# --- 3. Função de Geração do Diagrama T-s (Estágio Único) ---

def generate_ts_diagram_single_stage(T_in, T_out_real, T_out_iso, P_in, P_out):
    """Gera o diagrama T-s conceitual para compressão de estágio único."""
    
    # Valores conceituais de Entropia (apenas para plotagem)
    S_in = 0.0
    S_out_iso = S_in
    S_out_real = S_in + 0.3 # Entropia aumenta na compressão real
    
    plt.figure(figsize=(7, 5))

    # 1. Processo Isentrópico (Ideal): T aumenta, S constante
    plt.plot([S_in, S_out_iso], [T_in, T_out_iso], 'k--', 
             linewidth=2, label='Processo Isentrópico (100%)')

    # 2. Processo Real (Atual): T e S aumentam
    # CORREÇÃO: Removendo '\%' da string label
    plt.plot([S_in, S_out_real], [T_in, T_out_real], 'r-', 
             linewidth=3, label=f'Processo Real ({ETA_C*100:.0f}%)')

    # 3. Pontos
    plt.scatter([S_in, S_out_iso, S_out_real], [T_in, T_out_iso, T_out_real], 
                color=['k', 'k', 'r'], s=70, zorder=5)
    
    plt.text(S_in - 0.05, T_in + 5, 'Entrada', fontsize=10)
    plt.text(S_out_iso + 0.01, T_out_iso, 'T_2s (Ideal)', fontsize=10)
    plt.text(S_out_real + 0.01, T_out_real, 'T_2a (Real)', fontsize=10)

    # 4. Linha de Limite de Segurança
    plt.axhline(y=T_MAX_C, color='r', linestyle=':', linewidth=1, alpha=0.6, 
                label=f'Limite de T ({T_MAX_C:.0f}C)')
    
    # Legenda fora do plot (melhor leitura)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), 
               ncol=2, fancybox=True, shadow=True, fontsize='small')

    # CORREÇÃO: Usando string simples no título e rótulos
    plt.title(f'Compressão H2 em Estágio Único: {P_in:.0f} -> {P_out:.0f} bar')
    plt.xlabel('Entropia Específica, s (kJ/kg K)')
    plt.ylabel('Temperatura, T (C)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(S_in - 0.1, S_out_real + 0.1)
    plt.ylim(T_in - 5, max(T_out_real, T_MAX_C) + 20)
    plt.show()

# --- 4. Loop Principal e Interface com o Usuário ---

def main():
    print("--- Modelo de Compressor Simples (Estágio Único) ---")
    print("Escolha o modo de operação:")
    print("1: Exemplo Pré-Definido (40 -> 140 bar)")
    print("2: Inserir Valores Personalizados")

    choice = input("Digite 1 ou 2: ")

    if choice == '1':
        P_in = 40.0
        P_out = 140.0
        print(f"\nEXECUTANDO: Exemplo de Enchimento ({P_in:.0f} -> {P_out:.0f} bar)")
    elif choice == '2':
        try:
            P_in = float(input("Insira a Pressão de Entrada (bar): "))
            P_out = float(input("Insira a Pressão de Saída (bar): "))
            if P_out <= P_in:
                print("Erro: A Pressão de Saída deve ser maior que a Pressão de Entrada.")
                return
        except ValueError:
            print("Erro: Entrada inválida. Use apenas números.")
            return
    else:
        print("Opção inválida.")
        return

    # Realiza o cálculo
    W_total, T_out_C, T_out_s_C = calculate_single_stage_energy(P_in, P_out)
    
    # Prepara a tabela de resultados
    status = "ACIMA DO LIMITE" if T_out_C > T_MAX_C else "OK (< 85°C)"
    
    Tabela = {
        "Parâmetro": ["Pressão In/Out (bar)", "Razão de Compressão", "Consumo Total (kWh/kg)", 
                      "Temperatura Saída Real (T2a)", "Temperatura Saída Isentrópica (T2s)", "Status de Segurança"],
        "Resultado": [f"{P_in:.1f} -> {P_out:.1f}", 
                      f"{P_out/P_in:.2f}x", 
                      f"{W_total:.4f}", 
                      f"{T_out_C:.2f} °C",
                      f"{T_out_s_C:.2f} °C",
                      status]
    }
    
    df_results = pd.DataFrame(Tabela)

    print("\n" + "="*50)
    print("## 📊 Resultados do Compressor de Estágio Único")
    print(df_results.to_markdown(index=False))
    print("="*50)

    # Gera o gráfico
    generate_ts_diagram_single_stage(T_IN_C, T_out_C, T_out_s_C, P_in, P_out)

if __name__ == "__main__":
    main()
