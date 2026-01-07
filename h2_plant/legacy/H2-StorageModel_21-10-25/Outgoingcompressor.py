
from CoolProp import CoolProp as CP


def energy_required_for_outgoing_1_2stages(pressure, isentropic_efficiency, mechanical_efficiency, COP_cooling, outgoing_mass):

    # Part 1

    P_in_1 = pressure
    P_out_1 = 200e5  # Pa (500 bar)

    T_in = 273.15 + 10  # K (10 C)

    CR_total = P_out_1 / P_in_1

    number_of_stages = 2

    CR_per_stage = CR_total ** (1 / number_of_stages)

    outlet_pressure_stage_1 = P_in_1 * CR_per_stage
    outlet_pressure_stage_2 = outlet_pressure_stage_1 * CR_per_stage


    # Stage 1 calculations

    h_in_stage_1 = CP.PropsSI('H', 'P', P_in_1, 'T', T_in, 'Hydrogen')
    s_in_stage_1 = CP.PropsSI('S', 'P', P_in_1, 'T', T_in, 'Hydrogen')

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

    energy_required_outgoing_per_tank = ((total_work + total_cooling_work) * outgoing_mass) / 1e6  # MJ/min

    heat_generated_outgoing_per_tank_1 = q_cooling_stage_1 + q_cooling_stage_2

    return energy_required_outgoing_per_tank, heat_generated_outgoing_per_tank_1    


def energy_required_for_outgoing_1_3stages(pressure, isentropic_efficiency, mechanical_efficiency, COP_cooling, outgoing_mass):
     
    # Part 1

    P_in_1 = pressure
    P_out_1 = 200e5  # Pa (500 bar)

    T_in = 273.15 + 10  # K (10 C)

    CR_total = P_out_1 / P_in_1

    number_of_stages = 3

    CR_per_stage = CR_total ** (1 / number_of_stages)

    outlet_pressure_stage_1 = P_in_1 * CR_per_stage
    outlet_pressure_stage_2 = outlet_pressure_stage_1 * CR_per_stage
    outlet_pressure_stage_3 = outlet_pressure_stage_2 * CR_per_stage

    # Stage 1 calculations

    h_in_stage_1 = CP.PropsSI('H', 'P', P_in_1, 'T', T_in, 'Hydrogen')
    s_in_stage_1 = CP.PropsSI('S', 'P', P_in_1, 'T', T_in, 'Hydrogen')

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


    # Stage 3 calculations

    h_in_stage_3 = h_cooled_stage_2
    s_in_stage_3 = CP.PropsSI('S', 'P', outlet_pressure_stage_2, 'T', T_in, 'Hydrogen')

    h_out_s_stage_3 = CP.PropsSI('H', 'P', outlet_pressure_stage_3, 'S', s_in_stage_3, 'Hydrogen')
    h_out_stage_3 = h_in_stage_3 + (h_out_s_stage_3 - h_in_stage_3) / isentropic_efficiency


    work_stage_3 = h_out_stage_3 - h_in_stage_3

    h_cooled_stage_3 = CP.PropsSI('H', 'P', outlet_pressure_stage_3, 'T', T_in, 'Hydrogen')

    q_cooling_stage_3 = h_out_stage_3 - h_cooled_stage_3

    w_cooling_stage_3 = q_cooling_stage_3 / COP_cooling


    # Total work and cooling energy

    total_work = (work_stage_1 + work_stage_2 + work_stage_3) / mechanical_efficiency

    total_cooling_work = w_cooling_stage_1 + w_cooling_stage_2 + w_cooling_stage_3

    energy_required_outgoing_per_tank = ((total_work + total_cooling_work) * outgoing_mass) / 1e6  # MJ/min

    heat_generated_outgoing_per_tank_1 = q_cooling_stage_1 + q_cooling_stage_2 + q_cooling_stage_3

    return energy_required_outgoing_per_tank, heat_generated_outgoing_per_tank_1    




def energy_required_for_outgoing_2(isentropic_efficiency, mechanical_efficiency, COP_cooling, total_outgoing_mass, 
                                   out_h_200bar_10C, out_h_200_to_500bar_stage_1_out_s, out_h_200_to_500bar_stage_1_cooled, out_h_500bar_stage_2_out_s, out_h_500bar_stage_2_cooled):

    P_in = 200e5  # Pa (200 bar)
    P_out = 500e5  # Pa (500 bar)


    T_in = 273.15 + 10  # K (10 C)

    CR_total = P_out / P_in

    number_of_stages = 2

    CR_per_stage = CR_total ** (1 / number_of_stages)

    outlet_pressure_stage_1 = P_in * CR_per_stage
    outlet_pressure_stage_2 = outlet_pressure_stage_1 * CR_per_stage

    # Stage 1 calculations

    #h_in_stage_1 = CP.PropsSI('H', 'P', P_in, 'T', T_in, 'Hydrogen')
    #s_in_stage_1 = CP.PropsSI('S', 'P', P_in, 'T', T_in, 'Hydrogen')

    h_in_stage_1 = out_h_200bar_10C
    #s_in_stage_1 = out_s_200bar_10C


    #h_out_s_stage_1 = CP.PropsSI('H', 'P', outlet_pressure_stage_1, 'S', s_in_stage_1, 'Hydrogen')
    #h_out_stage_1 = h_in_stage_1 + (h_out_s_stage_1 - h_in_stage_1) / isentropic_efficiency

    h_out_s_stage_1 = out_h_200_to_500bar_stage_1_out_s
    h_out_stage_1 = h_in_stage_1 + (h_out_s_stage_1 - h_in_stage_1) / isentropic_efficiency


    work_stage_1 = h_out_stage_1 - h_in_stage_1

    #h_cooled_stage_1 = CP.PropsSI('H', 'P', outlet_pressure_stage_1, 'T', T_in, 'Hydrogen')
    h_cooled_stage_1 = out_h_200_to_500bar_stage_1_cooled

    q_cooling_stage_1 = h_out_stage_1 - h_cooled_stage_1

    w_cooling_stage_1 = q_cooling_stage_1 / COP_cooling

    # Stage 2 calculations

    h_in_stage_2 = h_cooled_stage_1
    #s_in_stage_2 = CP.PropsSI('S', 'P', outlet_pressure_stage_1, 'T', T_in, 'Hydrogen')
    #s_in_stage_2 = out_s_200_to_500bar_stage_2_in

    #h_out_s_stage_2 = CP.PropsSI('H', 'P', outlet_pressure_stage_2, 'S', s_in_stage_2, 'Hydrogen')

    h_out_s_stage_2 = out_h_500bar_stage_2_out_s
    h_out_stage_2 = h_in_stage_2 + (h_out_s_stage_2 - h_in_stage_2) / isentropic_efficiency

    work_stage_2 = h_out_stage_2 - h_in_stage_2

    #h_cooled_stage_2 = CP.PropsSI('H', 'P', outlet_pressure_stage_2, 'T', T_in, 'Hydrogen')
    h_cooled_stage_2 = out_h_500bar_stage_2_cooled

    q_cooling_stage_2 = h_out_stage_2 - h_cooled_stage_2

    w_cooling_stage_2 = q_cooling_stage_2 / COP_cooling

    # Total work and cooling energy

    total_work = (work_stage_1 + work_stage_2) / mechanical_efficiency

    total_cooling_work = w_cooling_stage_1 + w_cooling_stage_2

    energy_required_outgoing_2 = ((total_work + total_cooling_work) * total_outgoing_mass) / 1e6  # MJ/min 

    heat_generated_outgoing_per_tank_2 = q_cooling_stage_1 + q_cooling_stage_2

    return energy_required_outgoing_2, heat_generated_outgoing_per_tank_2   