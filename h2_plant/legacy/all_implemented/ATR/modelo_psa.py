# modelo_psa.py
def modelar_psa(m_dot_h2o_vap_in, P_in_bar):
    """Remove a umidade em fase vapor por adsorção."""
    m_dot_removida = m_dot_h2o_vap_in * 0.9999
    return {
        "m_dot_agua_removida_kg_s": m_dot_removida,
        "P_out_bar": P_in_bar - 0.3,
        "W_dot_kW": 0.85 # Potência de purga/regeneração
    }