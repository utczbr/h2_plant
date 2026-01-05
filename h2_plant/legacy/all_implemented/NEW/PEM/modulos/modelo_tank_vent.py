import numpy as np
import math

# ==============================================================================
# CLASSE: FlashDrumModel para Modelagem e Dimensionamento de Vasos de Desgaseificação
#         (AGORA: Tanque Equalizador com Ventilação)
# ==============================================================================

class FlashDrumModel:
    """
    Modelagem e Dimensionamento de um Tanque Equalizador de Água de Dreno,
    projetado com diâmetro fixo para alto fluxo e estimativa de dessorção
    baseada na Lei de Henry.
    """

    def __init__(self, T_C, P_op_kPa, Q_L_m3_h, C_gas_in_mol_L, eficiencia_desejada, gas_name, D_tanque_m):
        """
        Inicializa o modelo com os parâmetros de entrada, incluindo o diâmetro.

        Args:
            T_C (float): Temperatura de operação em °C.
            P_op_kPa (float): Pressão de operação do flash drum em kPa.
            Q_L_m3_h (float): Vazão volumétrica de líquido (água) em m³/h.
            C_gas_in_mol_L (float): Concentração inicial do gás na água em mol/L.
            eficiencia_desejada (float): Eficiência de remoção desejada (0 a 1).
            gas_name (str): Nome do gás (ex: 'O2', 'H2').
            D_tanque_m (float): Diâmetro do tanque em metros (NOVO).
        """
        self.T_C = T_C
        self.P_op_kPa = P_op_kPa
        self.Q_L_m3_h = Q_L_m3_h
        self.C_gas_in_mol_L = C_gas_in_mol_L
        self.eficiencia_desejada = eficiencia_desejada
        self.gas_name = gas_name

        # Correção Crítica 1: Converter a vazão volumétrica de líquido para m³/s (SI)
        self.Q_L_m3_s = self.Q_L_m3_h / 3600.0

        # --- NOVOS PARÂMETROS DE DIMENSIONAMENTO (Tanque Equalizador) ---
        self.D_tanque_m = D_tanque_m              # Diâmetro agora é passado como argumento
        self.tau_retencao_min = 1.0        # Tempo de retenção reduzido (1 minuto - Equalizador)
        self.NIVEL_LIQUIDO_PCT = 0.70      # Nível de saída de líquido (70% da altura total)
        # -----------------------------------------------------------------

        # Constantes (ajustáveis, dependentes da T e P)
        self.H_kPa_L_mol = {
            'O2': 79000.0,  
            'H2': 72000.0   
        }
        # Densidades (valores aproximados para 25°C)
        self.rho_L = 997.0 # Densidade da água (kg/m³)
        self.rho_V = 1.3  # Densidade do vapor (gás na fase vapor) (kg/m³) - Gás puro
        self.sigma = 0.072 # Tensão superficial da água (N/m)

        # Configuração de T e P para cálculos internos
        self.T_K = self.T_C + 273.15
        self.R = 8.314 # Constante universal dos gases (J/(mol·K))


    def modelar_remocao(self):
        """
        Realiza o cálculo de equilíbrio e balanço de massa para a remoção do gás.

        Returns:
            dict: Resultados da modelagem.
        """
        if self.C_gas_in_mol_L < 1e-10:
             return {
                'C_final_mol_L': 0.0,
                'P_parcial_req_kPa': 0.0,
                'Q_molar_removida_mol_h': 0.0,
                'Q_V_m3_h': 0.0,
                'eficiencia_realizada': 0.0
            }
            
        H = self.H_kPa_L_mol[self.gas_name]

        # 1. Concentração do Gás Dissolvido no Equilíbrio (Lei de Henry)
        C_final_mol_L = self.C_gas_in_mol_L * (1 - self.eficiencia_desejada)
        
        # 2. Pressão Parcial de Equilíbrio (pressão mínima requerida no topo)
        P_parcial_gas_req = C_final_mol_L * H
        
        # 3. Massa de Gás Removida (Balanço de Massa)
        # Vazão molar de líquido (assumindo 1000 L/m³ da água)
        Q_L_L_h = self.Q_L_m3_h * 1000 # L/h
        # Vazão molar de Gás Removida
        Q_molar_removida = (self.C_gas_in_mol_L - C_final_mol_L) * Q_L_L_h # mol/h
        
        # Vazão Volumétrica do Vapor Removido (idealmente, se fosse gás puro a P e T do vaso)
        # V = nRT/P (n = Q_molar_removida / 3600 (mol/s))
        Q_V_m3_s = (Q_molar_removida / 3600) * self.R * self.T_K / (self.P_op_kPa * 1000) # (m³/s) - P precisa estar em Pa
        Q_V_m3_h = Q_V_m3_s * 3600 # m³/h

        return {
            'C_final_mol_L': C_final_mol_L,
            'P_parcial_req_kPa': P_parcial_gas_req,
            'Q_molar_removida_mol_h': Q_molar_removida,
            'Q_V_m3_h': Q_V_m3_h,
            'eficiencia_realizada': (self.C_gas_in_mol_L - C_final_mol_L) / self.C_gas_in_mol_L
        }


    def dimensionar_vaso(self, resultados_modelagem):
        """
        Dimensiona o Tanque Equalizador com base no diâmetro fixo e o tempo de retenção (1 minuto).
        """
        
        # 1. Diâmetro do Vaso (D) - Valor Forçado
        D_m = self.D_tanque_m
        A_m2 = math.pi * D_m**2 / 4
        
        if A_m2 < 1e-10:
             return {
                'v_max_m_s': float('inf'),
                'D_m': D_m,
                'h_L_m': float('inf'),
                'H_m': float('inf'),
                'L_D_ratio': float('inf'),
                'tempo_retencao_min': self.tau_retencao_min
            }

        # 2. Altura de Retenção de Líquido (h_L)
        tau_s = self.tau_retencao_min * 60 # Tempo de retenção (em segundos)
        
        # Volume de líquido (m³)
        V_L_m3 = self.Q_L_m3_s * tau_s 
        
        # h_L = V_L / A 
        h_L_m = V_L_m3 / A_m2 
        
        # 3. Altura Total do Vaso (H)
        # Altura Total = Altura de Retenção / Nível de Saída (0.70)
        H_m = h_L_m / self.NIVEL_LIQUIDO_PCT
        
        # 4. Velocidade "Máxima" de Vapor (Apenas para comparação - não é critério)
        Q_V_m3_s = resultados_modelagem['Q_V_m3_h'] / 3600 # m³/s
        v_max_calc = Q_V_m3_s / A_m2 if Q_V_m3_s > 0 else 0.0

        # Relação Altura/Diâmetro (L/D)
        L_D_ratio = H_m / D_m 
        
        return {
            'v_max_m_s': v_max_calc,
            'D_m': D_m,
            'h_L_m': h_L_m,
            'H_m': H_m,
            'L_D_ratio': L_D_ratio,
            'tempo_retencao_min': self.tau_retencao_min
        }

    def simular(self):
        """Executa a modelagem e o dimensionamento."""
        modelagem = self.modelar_remocao()
        dimensionamento = self.dimensionar_vaso(modelagem)
        return modelagem, dimensionamento

# ==============================================================================
# EXECUÇÃO DA SIMULAÇÃO (EXEMPLO)
# ==============================================================================

# --- PARÂMETROS DE ENTRADA (MUDE AQUI) ---
T_C = 30.0                  # Temperatura de Operação (°C)
P_op_kPa = 101.325          # Pressão Atmosférica (kPa) - 1 atm
Q_L_m3_h = 50.0             # Vazão Volumétrica da Água de Dreno (m³/h) - Placeholder
eficiencia_desejada = 0.95  # Eficiência de Remoção Desejada (95%)

# Concentrações Iniciais (MUDE AQUI - Exemplo: 10 ppm em peso)
C_O2_in_mol_L = 0.00055 # Concentração de O2 na água (mol/L)
C_H2_in_mol_L = 0.00050 # Concentração de H2 na água (mol/L)


def imprimir_resultados(gas_name, modelagem, dimensionamento):
    """Formata e imprime os resultados no terminal."""
    print("="*60)
    print(f"       ✅ RESULTADOS DA SIMULAÇÃO: Tanque Equalizador de {gas_name}       ")
    print("="*60)
    
    # Modelo
    print("\n--- 📊 Modelagem (Remoção e Equilíbrio) ---")
    print(f"Eficiência Desejada:            {modelagem['eficiencia_realizada']:.2%}")
    print(f"Concentração de {gas_name} Final:    {modelagem['C_final_mol_L']:.4e} mol/L")
    print(f"Vazão de {gas_name} Removida:        {modelagem['Q_molar_removida_mol_h']:.2f} mol/h")
    print(f"Vazão Volumétrica de Vapor:     {modelagem['Q_V_m3_h']:.4f} m³/h")
    print(f"Pressão Parcial Requerida:      {modelagem['P_parcial_req_kPa']:.2f} kPa")
    
    # Dimensionamento (Valores ajustados e formatados)
    print("\n--- 📏 Dimensionamento (Tanque Equalizador/Ventilado) ---")
    print(f"Diâmetro do Tanque (D):         {dimensionamento['D_m']:.2f} m (FIXO)")
    print(f"Tempo de Retenção de Líquido:   {dimensionamento['tempo_retencao_min']:.1f} min (ALVO)")
    print(f"Altura de Retenção de Líquido:  {dimensionamento['h_L_m']:.2f} m")
    print(f"Altura Total do Vaso (H):       {dimensionamento['H_m']:.2f} m")
    print(f"Razão Altura/Diâmetro (L/D):    {dimensionamento['L_D_ratio']:.2f}")
    print(f"Velocidade de Vapor Calculada:  {dimensionamento['v_max_m_s']:.4f} m/s")
    print("="*60)
    print("\n")


# 1. Simulação para a corrente de OXIGÊNIO (O2) - D = 1.5 m
o2_model = FlashDrumModel(
    T_C=T_C, 
    P_op_kPa=P_op_kPa, 
    Q_L_m3_h=Q_L_m3_h, 
    C_gas_in_mol_L=C_O2_in_mol_L, 
    eficiencia_desejada=eficiencia_desejada, 
    gas_name='O2',
    D_tanque_m=1.5 # Diâmetro para o dreno de O2
)
modelagem_o2, dimensionamento_o2 = o2_model.simular()
imprimir_resultados('Oxigênio (O2)', modelagem_o2, dimensionamento_o2)


# 2. Simulação para a corrente de HIDROGÊNIO (H2) - D = 1.0 m
h2_model = FlashDrumModel(
    T_C=T_C, 
    P_op_kPa=P_op_kPa, 
    Q_L_m3_h=Q_L_m3_h, 
    C_gas_in_mol_L=C_H2_in_mol_L, 
    eficiencia_desejada=eficiencia_desejada, 
    gas_name='H2',
    D_tanque_m=1.0 # Diâmetro para o dreno de H2
)
modelagem_h2, dimensionamento_h2 = h2_model.simular()
imprimir_resultados('Hidrogênio (H2)', modelagem_h2, dimensionamento_h2)