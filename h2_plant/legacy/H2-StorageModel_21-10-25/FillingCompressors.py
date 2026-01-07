
from CoolProp import CoolProp as CP


def energy_requirement_for_filling(production_pressure, filling_pressure, T_production, isentropic_efficiency, mechanical_efficiency, COP_cooling, actual_hydrogen_production, fill_h_40bar_10C, fill_s_40bar_10C):

    P_in = production_pressure
    P_out = filling_pressure

    CR_total = P_out / P_in

    number_of_stages = 2

    CR_stage = CR_total ** (1 / number_of_stages)


    outlet_pressure_stage_1 = P_in * CR_stage
    outlet_pressure_stage_2 = outlet_pressure_stage_1 * CR_stage

    T_in = T_production

    hydrogen_mass_flow_rate = actual_hydrogen_production  # kg/min


    # Determining the energy requirements for each stage
    # These include the work done by the compressor and the cooling energy required

    # First, the enthalpy and entropy at the inlet are calculated with CoolProp
    # Then the enthalpy at the outlet is calculated using the isentropic process (so ingoing entropy and outlet pressure)

    # Then, the real enthalpy at the outlet is calculated using the isentropic efficiency
    # The temperature at the outlet is calculated using the enthalpy and pressure

    # The work done in each stage is the difference between the inlet and outlet enthalpies
    # The cooling energy is calculated based on the enthalpy difference at the outlet and the inlet temperature



    # Stage 1 calculations

    #h_in_stage_1 = CP.PropsSI('H', 'P', P_in, 'T', T_in, 'Hydrogen')
    #s_in_stage_1 = CP.PropsSI('S', 'P', P_in, 'T', T_in, 'Hydrogen')
    h_in_stage_1 = fill_h_40bar_10C
    s_in_stage_1 = fill_s_40bar_10C

    h_out_s_stage_1 = CP.PropsSI('H', 'P', outlet_pressure_stage_1, 'S', s_in_stage_1, 'Hydrogen')
    h_out_stage_1 = h_in_stage_1 + (h_out_s_stage_1 - h_in_stage_1) / isentropic_efficiency


    work_stage_1 = h_out_stage_1 - h_in_stage_1

    h_cooled_stage_1 = CP.PropsSI('H', 'P', outlet_pressure_stage_1, 'T', T_in, 'Hydrogen')

    q_cooling_stage_1 = h_out_stage_1 - h_cooled_stage_1

    w_cooling_stage_1 = q_cooling_stage_1 / COP_cooling


    # Stage 2 calculations

    h_in_stage_2 = h_cooled_stage_1
    s_in_stage_2 = CP.PropsSI('S', 'P', outlet_pressure_stage_1, 'T', T_in, 'Hydrogen')

    h_out_s_stage_2 = CP.PropsSI('H', 'P', outlet_pressure_stage_2, 'S', s_in_stage_2, 'Hydrogen')
    h_out_stage_2 = h_in_stage_2 + (h_out_s_stage_2 - h_in_stage_2) / isentropic_efficiency

    work_stage_2 = h_out_stage_2 - h_in_stage_2

    h_cooled_stage_2 = CP.PropsSI('H', 'P', outlet_pressure_stage_2, 'T', T_in, 'Hydrogen')

    q_cooling_stage_2 = h_out_stage_2 - h_cooled_stage_2

    w_cooling_stage_2 = q_cooling_stage_2 / COP_cooling



    # Total work and cooling energy

    total_work = (work_stage_1 + work_stage_2) / mechanical_efficiency

    total_cooling_work = w_cooling_stage_1 + w_cooling_stage_2 

    total_energy_required_for_filling = total_work + total_cooling_work

    heat_generated_filling = q_cooling_stage_1 + q_cooling_stage_2

    energy_required_filling = (total_energy_required_for_filling * hydrogen_mass_flow_rate) / 1e6  # MJ/min

    return energy_required_filling, heat_generated_filling







    