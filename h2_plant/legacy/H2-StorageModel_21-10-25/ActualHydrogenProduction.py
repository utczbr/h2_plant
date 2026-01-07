
def calculate_hydrogen_production(wind_power, number_of_wind_turbines, i, max_power_output, molar_mass_hydrogen):

    actual_wind_power = wind_power * number_of_wind_turbines

    # Constants

    F = 96485;         # Faraday constant (C/mol)
    z = 4;             # Electrons per O2 molecule
    eta_f = 1;         # Faraday efficiency
    Parcell = 85;      # Number of cells per stack
    Parstack = 36;     # Number of parallel stacks
    Aeff = 0.03;       # Effective area of one cell (m^2/cell)
    #molar_mass_hydrogen_kg_per_mol = CP.PropsSI('M', 'Hydrogen')  # Molar mass of hydrogen in kg/mol
    molar_mass_hydrogen_kg_per_kmol = molar_mass_hydrogen * 1000  # Convert to kg/kmol


    current_10_minutes = i // 10

    current_wind_power = actual_wind_power[current_10_minutes]


    if current_wind_power >= max_power_output:
        current_wind_power = max_power_output  # Limit to maximum power output, all the above is send to the grid

    if current_wind_power > 11520:
        current_wind_power = current_wind_power - 11520 # 11520 kW is the power required for the SOEC, remaining power will be used for PEM. If the wind power is less than 11520 kW, all the power will be used for PEM.
        SOEC_hydrogen_output = 5.12 # kg/min (SOEC hydrogen production rate at 80% of itscapacity)


    else:
        SOEC_hydrogen_output = 0  # kg/min

    power_stack = current_wind_power / 36

    icell_mAcm2 = -0.026 * power_stack**2 + 23.363 * power_stack + 30.357

    # Convert current density from mA/cm² to A/m²

    icell_Am2 = icell_mAcm2 * 10

    # Calculate the total current (A)
    I_total = Parcell * Parstack * icell_Am2 * Aeff * eta_f

    # Use Faraday's law to calculate the oxygen production rate in mol/s

    n_dot_O2_gen = I_total / (z * F)

    #Convert oxygen production rate to kmol/hr

    productionRate_kmolhr = n_dot_O2_gen * 3.6

    hydrogen_production_molar = productionRate_kmolhr * 2  # 2 moles of H2 for every mole of O2

    hydrogen_production_kg_hr = hydrogen_production_molar * molar_mass_hydrogen_kg_per_kmol  # Convert to kg/hr

    hydrogen_production_PEM_kg = hydrogen_production_kg_hr / 60 # Convert from kg/hr to kg (kg per minute)

    hydrogen_production_potential = hydrogen_production_PEM_kg + SOEC_hydrogen_output  # kg/min

    return hydrogen_production_potential
    


