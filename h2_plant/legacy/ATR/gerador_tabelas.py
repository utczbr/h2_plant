# gerador_tabelas.py
import pandas as pd

def imprimir_tabelas(df):
    # Cálculos de erro conforme original
    df['Err_H08'] = (abs(df['Q_H08'] - df['Q_H08_plan']) / df['Q_H08_plan']) * 100
    df['Err_H09'] = (abs(df['Q_H09'] - df['Q_H09_plan']) / df['Q_H09_plan']) * 100
    
    df_view = pd.concat([df.head(5), df.tail(5)]).copy()

    # TABELAS ORIGINAIS (MANTIDAS SEM ALTERAÇÃO)
    headers1 = {
        'F_O2': 'F_O2',
        'T_out_H08': 'T_out_H08 (°C)', 'Q_H08': 'Q_H08 (kW)', 'Err_H08': 'Err_H08 (%)',
        'T_out_H09': 'T_out_H09 (°C)', 'Q_H09': 'Q_H09 (kW)', 'Err_H09': 'Err_H09 (%)'
    }

    headers2 = {
        'F_O2': 'F_O2',
        'T_in_HEX': 'T_in_HEX (°C)', 
        'T_out_HEX': 'T_out_HEX (°C)', 
        'Q_HEX': 'Q_HEX (kW)',
        'T_in_H2O_HEX': 'T_in_H2O (°C)', 
        'F_H2O_HEX': 'F_H2O (kg/s)', 
        'T_out_H2O_HEX': 'T_out_H2O (°C)',
        'T_out_Chil': 'T_out_Chil (°C)', 
        'Q_Chil': 'Q_Chil (kW)'
    }
    
    headers3 = {
        'F_O2': 'F_O2',
        'H2O_liq_HEX_out': 'Liq_Pos_HEX (kg/h)', 
        'H2O_vap_HEX_out': 'Vap_Pos_HEX (kg/h)',
        'H2O_rem_Cic1': 'Rem_Cic1 (kg/h)', 
        'Liq_Chil': 'Liq_Chil (kg/h)',
        'H2O_rem_Cic2': 'Rem_Cic2 (kg/h)', 
        'H2O_rem_Coal': 'Rem_Coal (kg/h)', 
        'H2O_rem_PSA': 'Rem_PSA (kg/h)'
    }
    
    headers4 = {
        'F_O2': 'F_O2',
        'W_Chil': 'W_Chil (kW)', 'W_Coal': 'W_Coal (kW)', 'W_PSA': 'W_PSA (kW)',
        'P_Fin': 'P_Fin (bar)', 'T_Fin': 'T_Fin (°C)', 'H2O_PPM_Final': 'H2O_Final (PPM)'
    }

    headers5 = {
        'F_O2': 'F_O2',
        'W_Chil': 'Consumo Chiller (kW)',
        'W_Coal': 'Consumo Coal (kW)',
        'W_PSA': 'Consumo PSA (kW)',
        'W_Total': 'CONSUMO TOTAL (kW)'
    }

    headers6 = {
        'F_O2': 'F_O2',
        'P_PSA_Out': 'P_In_C1 (bar)',
        'T_C1_Out': 'T_Out_C1 (°C)',
        'P_Inter': 'P_Inter (bar)',
        'T_IC_Out': 'T_Out_IC (°C)',
        'T_C2_Out': 'T_Out_C2 (°C)',
        'P_Fin': 'P_Final (bar)'
    }

    headers7 = {
        'F_O2': 'F_O2',
        'F_H2O_Recup': 'Vazão Recuperada (kg/h)',
        'T_H2O_Recup': 'Temp. Recuperada (°C)'
    }

    headers8 = {
        'F_O2': 'F_O2',
        'F_H2O_Recup': 'Água Reciclo (kg/h)',
        'F_H2O_Makeup': 'Água Reposição (kg/h)',
        'F_H2O_ATR_Final': 'Total para ATR (kg/h)',
        'F_H2O_Target_Plan': 'Alvo Planilha (kg/h)',
        'T_H2O_ATR_Final': 'Temp. Mistura Final (°C)'
    }

    # NOVA TABELA 9: SOLICITADA PELO USUÁRIO
    headers9 = {
        'F_O2': 'F_O2',
        'T_Boiler_In': 'Gas_In_Boiler (°C)',
        'T_Boiler_Out': 'Gas_Out_Boiler (°C)',
        'T_in_H2O_HEX': 'T_In_H2O_HEX (°C)',
        'F_H2O_HEX': 'F_H2O_HEX (kg/s)',
        'T_out_H2O_HEX': 'T_Out_H2O_HEX (°C)'
    }

    largura_linha = 200

    def print_tab(titulo, headers):
        print("\n" + "="*largura_linha)
        print(titulo)
        print("-" * largura_linha)
        cols = [c for c in headers.keys() if c in df_view.columns]
        print(df_view[cols].rename(columns=headers).to_string(index=False, float_format=lambda x: "{:,.2f}".format(x)))

    # Chamadas originais
    print_tab("TABELA 1: VALIDAÇÃO DE MODELO E DESEMPENHO TÉRMICO (H08, H09)", headers1)
    print_tab("TABELA 2: EQUIPAMENTOS DE TROCA TÉRMICA (HEX, CHILLER)", headers2)
    print_tab("TABELA 3: BALANÇO DE ÁGUA E ESTADO DE FASES (kg/h)", headers3)
    print_tab("TABELA 4: UTILIDADES, POTÊNCIA E ESTADO FINAL", headers4)
    print_tab("TABELA 5: CONSUMO DE POTÊNCIA DETALHADO (kW)", headers5)
    print_tab("TABELA 6: TREM DE COMPRESSÃO - EVOLUÇÃO DE PRESSÃO E TEMPERATURA", headers6)
    print_tab("TABELA 7: MISTURA DE CONDENSADOS (RECUPERAÇÃO CICLONES)", headers7)
    print_tab("TABELA 8: BALANÇO DE MASSA E RECICLO PARA O ATR", headers8)
    
    # Nova chamada
    print_tab("TABELA 9: DESEMPENHO AQUECIMENTO GÁS (BOILER) E ÁGUA HEX", headers9)
    
    print("="*largura_linha + "\n")