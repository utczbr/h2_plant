# modelo_coalescedor.py
def modelar_coalescedor(m_dot_h2o_liq_in, eficiencia=0.98):
    """Remove fisicamente o condensado líquido residual."""
    m_dot_removida = m_dot_h2o_liq_in * eficiencia
    return {
        "m_dot_agua_removida_kg_s": m_dot_removida,
        "W_dot_kW": 0.05 # Potência nominal de drenagem
    }