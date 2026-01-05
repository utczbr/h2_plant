from CoolProp.CoolProp import PropsSI

def modelar_ciclone_separador(
    gas_fluido: str,
    m_dot_total: float,
    P_bar: float,
    T_C: float,
    eficiencia_separacao: float = 0.99
) -> dict:
    """
    Modela um ciclone separador que remove uma porcentagem da fase líquida.
    Utiliza CoolProp para determinar a qualidade do vapor (Q).
    """
    P_pa = P_bar * 1e5
    T_k = T_C + 273.15

    try:
        # Título (vapor quality): 1 = vapor saturado, 0 = líquido saturado
        # Se T > T_crit ou P > P_crit, o CoolProp lidará conforme a fase
        vapor_quality = PropsSI('Q', 'T', T_k, 'P', P_pa, 'Water')
        
        # Se vapor_quality for 1.0, não há líquido.
        # Se for entre 0 e 1, a fração de líquido é (1 - Q)
        friz_liquida = 1.0 - max(0, min(1, vapor_quality))
    except:
        # Caso esteja fora da zona de saturação (superaquecido), assume-se 0 líquido
        friz_liquida = 0.0

    m_dot_liquido_total = m_dot_total * friz_liquida
    m_dot_removido = m_dot_liquido_total * eficiencia_separacao
    m_dot_saida = m_dot_total - m_dot_removido

    return {
        "m_dot_saida_kg_s": m_dot_saida,
        "m_dot_agua_removida_kg_s": m_dot_removido,
        "m_dot_agua_removida_kg_h": m_dot_removido * 3600.0
    }