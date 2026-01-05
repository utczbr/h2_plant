# modelo_dry_cooler.py
from CoolProp.CoolProp import PropsSI

def modelar_resfriador_condensador(m_dot_gas, m_dot_h2o, T_in, T_out, P_bar, cp_mix, considerar_latente=False):
    """
    Modelo de resfriador reintegrado com a lógica de alta aderência.
    Utiliza o cp_mix validado sobre a massa de gás para espelhar os resultados do Aspen.
    """
    P_pa = P_bar * 1e5
    T_k_out = T_out + 273.15
    
    # Na versão de alta aderência, Q é calculado sobre m_dot_gas 
    # pois o cp_mix já contempla a energia da mistura úmida.
    Q_total = m_dot_gas * cp_mix * (T_in - T_out)
    
    # Inventário de condensado para os sistemas sequentes (Ciclone/Chiller)
    try:
        # Título de vapor (Q): 1 = vapor, 0 = líquido
        q_vapor = PropsSI('Q', 'T', T_k_out, 'P', P_pa, 'Water')
        m_dot_liq = m_dot_h2o * (1.0 - max(0, min(1, q_vapor)))
    except:
        # Se superaquecido, m_dot_liq = 0
        m_dot_liq = 0.0

    return {
        "Q_total_kW": Q_total,
        "m_dot_h2o_liq_no_fluxo": m_dot_liq
    }