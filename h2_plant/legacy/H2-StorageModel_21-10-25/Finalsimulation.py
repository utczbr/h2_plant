# Main simulation script for the hydrogen storage model

# Important notes when changing parameters:
# - When changing the minimum pressure:
#   - Change the number of stages for the emptying compressors in HPProcessLogicNoLP.py (2 stages for 70 bar, 3 stages for < 70 bar)
# - When changing the maximum pressure:
#   - Change the number of stages for the filling compressors in the main simulation loop (2 stages for 100 bar, 3 stages for > 100 bar) (in emptying_tanks function)
# - When changing the number of operational days:
#   - Change the beginning of the weekend value (5 OD: Friday 23:00, 6 OD: Saturday 23:00)
#   - Change night shift values in H2DemandwithNightShift.py (get_night_shift_demand function)
#   - When changing to 7 OD, change to weekend = False, and set a high value for time_until_weekend



# These are the imports required for the simulation
# Importing the filling station code
from H2DemandwithNightShift import FillingStations

# Importing the energy price function and state classification
from EnergyPriceFunction import load_energy_price_data
from EnergyPriceState import classify_energy_price, filling_thresholds, emptying_thresholds

# Choose one of the following imports for the filling compressors, based on the pressure requirements
# For a max pressure of 100 bar (2 stages required), everything higher will need 3 stages
from FillingCompressors import energy_requirement_for_filling
from FillingCompressors3stages import filling_compressors_3stages

# Importing the control logic for the storage tanks
from HPProcessLogicNoLP import HP_Tanks

# Importing the hydrogen production calculation function
from ActualHydrogenProduction import calculate_hydrogen_production

# Importing necessary libraries for data handling and plotting
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd



# Temperature and molar mass of hydrogen
T = 273.15 + 10
molar_mass_hydrogen = CP.PropsSI('M', 'Hydrogen')  # Molar mass of hydrogen in kg/mol


# simulation parameters
simulation_hours = 8760  # 90 days of simulation
simulation_minutes = simulation_hours * 60


# Hydrogen production values
number_of_wind_turbines = 12
max_power_output = 20000  # kW (maximum power output of the wind turbines) (If it is higher than 15000 kW, the excess power is sent to the grid)
wind_speed = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
power_output = [0, 10, 330, 560, 1000, 1470, 1980, 2800, 4000, 4800, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
wind_data = pd.read_excel('wind_data.xlsx')
wind_power = np.interp(wind_data['Wind Speed'], wind_speed, power_output)


# energy price data
price_array, low_to_medium_threshold, medium_to_high_threshold = load_energy_price_data()


# Compressor parameters
production_pressure = 40e5  # Pa (40 bar)
T_production = T  # K (outlet temperature of the production process)
isentropic_efficiency = 0.65  # Efficiency of the compressor
mechanical_efficiency = 0.9  # Efficiency of the compressor motor
COP_cooling = 3.0 # Coefficient of performance for cooling (assumed value)


# Compressor values (Having these values outside the loop to avoid recalculating them every time)

# For filling
fill_h_40bar_10C = CP.PropsSI('H', 'P', production_pressure, 'T', T, 'Hydrogen')  # Enthalpy at 40 bar and 10 C
fill_s_40bar_10C = CP.PropsSI('S', 'P', production_pressure, 'T', T, 'Hydrogen')  # Entropy at 40 bar and 10 C

# For outgoing
# 316.22 bar is the pressure after the first stage of compression

out_h_200bar_10C = CP.PropsSI('H', 'P', 200e5, 'T', T, 'Hydrogen')  # Enthalpy at 200 bar and 10 C
out_s_200bar_10C = CP.PropsSI('S', 'P', 200e5, 'T', T, 'Hydrogen')  # Entropy at 200 bar and 10 C

out_h_200_to_500bar_stage_1_out_s = CP.PropsSI('H', 'P', 316.22776601683796e5, 'S', out_s_200bar_10C, 'Hydrogen')
out_h_200_to_500bar_stage_1_cooled = CP.PropsSI('H', 'P', 316.22776601683796e5, 'T', T, 'Hydrogen')

out_s_200_to_500bar_stage_2_in = CP.PropsSI('S', 'P', 316.22776601683796e5, 'T', T, 'Hydrogen')
out_h_500bar_stage_2_out_s = CP.PropsSI('H', 'P', 500e5, 'S', out_s_200_to_500bar_stage_2_in, 'Hydrogen')

out_h_500bar_stage_2_cooled = CP.PropsSI('H', 'P', 500e5, 'T', T, 'Hydrogen')




# Filling stations parameters
number_of_filling_stations = 5
simultaneously_filling_tanks = 5  # Number of tanks that can be filled simultaneously
simultaneously_emptying_tanks = 5  # Number of tanks that can be emptied simultaneously
tanks_filling_during_startup = 10  # Number of tanks that are filled during startup

truck_capacity = 280  # kg (capacity of a truck)
max_flowrate = 90 / 60  # kg/s (maximum flow rate for filling stations)
min_flowrate = 27 / 60  # kg/s (minimum flow rate for filling stations) (30% of max flowrate)

  
# Storage tank parameters
HP_total_volume = 3221 # m^3 (total volume of the high-pressure tanks) 
HP_max_pressure = 10e6  # Pa (maximum pressure of the high-pressure tanks)
HP_min_pressure = 7e6  # Pa (minimum pressure of the high-pressure tanks) 

number_of_tanks = 30 # Number of high-pressure tanks/compartments
volume_per_tank = HP_total_volume / number_of_tanks  # m^3 (volume of each high-pressure tank/compartment)

density_at_min_pressure = CP.PropsSI('D', 'P', HP_min_pressure, 'T', T, 'Hydrogen')  # kg/m³
density_at_max_pressure = CP.PropsSI('D', 'P', HP_max_pressure, 'T', T, 'Hydrogen')  # kg/m³

cushion_mass_per_tank = density_at_min_pressure * volume_per_tank  # kg (cushion gas mass in each high-pressure tank/compartment)
total_mass_per_tank = density_at_max_pressure * volume_per_tank  # kg (total mass of hydrogen at maximum pressure in each high-pressure tank/compartment)
HP_tank_capacity = total_mass_per_tank - cushion_mass_per_tank  # kg (capacity of each high-pressure tank/compartment)
total_capacity_of_storage = HP_tank_capacity * number_of_tanks  # kg (total capacity of all high-pressure tanks/compartments)



# Parameters for changing the demand (changing the cooldown between trucks) (NOT BEING USED CURRENTLY!)
minimum_fill_time = truck_capacity / max_flowrate  # minutes (minimum time to fill a truck at maximum flow rate)
demand_ratio = 1
cooldown_time = minimum_fill_time / demand_ratio



# Set up the high-pressure tanks and filling stations
hp_tanks = HP_Tanks(number_of_tanks)
filling_stations = [FillingStations(i, cooldown_time) for i in range(number_of_filling_stations)]


# Initial values
overall_hydrogen_demand = 0  # kg (the hydrogen demand of all filling stations combined at each time step)
total_reserved_hydrogen = 0  # kg (the hydrogen reserved for each filling station at each time step)

cumulative_hydrogen_production_potential = 0  # kg (cumulative hydrogen production)
cumulative_actual_hydrogen_production = 0  # kg (cumulative actual hydrogen production after unhandled)
cumulative_hydrogen_demand = 0  # kg (cumulative hydrogen demand)

total_unhandled_hydrogen = 0  # kg (total unhandled hydrogen)
total_unhandled_demand = 0  # kg (total unhandled demand)

total_energy_required_filling = 0  # kWh (total energy required for compression and filling)
total_energy_cost_filling = 0  # euros (total energy cost)

total_energy_required_outgoing = 0  # kWh (total energy required for outgoing hydrogen)
overall_total_energy_required_outgoing = 0  # kWh (total energy required for outgoing hydrogen)
total_energy_cost_outgoing = 0  # euros (total energy cost for outgoing hydrogen)
total_heat_generated_outgoing = 0  # MJ (total heat generated from outgoing compressors)
heat_generated_filling = 0  # MJ (heat generated from filling compressors)

cumulative_heat_generated = 0  # MJ (cumulative heat generated from compressors)

startup = True # Initial state of the system (True means startup phase)


# storing the results for plotting
time_steps = []
production_values = []
actual_production_values = []
demand_values = []
#HP_tanks_mass_values = {i: [] for i in range(len(hp_tanks.tanks))} 
HP_total_mass_values = []
energy_price_over_time = []



# Main simulation loop

for i in range(simulation_minutes):

    # Checking current time
    current_time = i
    current_10_minutes = i // 10
    current_hour = i // 60  

    # Checking if weekend or not
    # Assuming the week starts on Monday at 00:00 (0 minutes)
    minutes_per_week = 7 * 24 * 60                                                                                          
    minutes_begin_weekend = 5 * 24 * 60 + 23 * 60 # Friday 23:00                                                                                    
    minutes_end_weekend = 7 * 60 # Monday 7:00           

    time_in_week = current_time % minutes_per_week  


    # If weekend operation is considered, choose the right line!!!

    weekend = (time_in_week >= minutes_begin_weekend) or (time_in_week <minutes_end_weekend)
    #weekend = False # Weekend operation is not considered in this model

    
    # Same with these lines, choose the right one based on weekend operation consideration

    if time_in_week < minutes_begin_weekend:
        time_until_weekend = minutes_begin_weekend - time_in_week
    else:
        time_until_weekend = (minutes_per_week - time_in_week) + minutes_begin_weekend

    #time_until_weekend = 1000000  # If weekend operation is not considered, set a high value


    # Get the current energy price
    current_price = price_array[current_hour]

    # Classify energy price state (Low, Medium, High) (NOT USED CURRENTLY!)
    energy_price_state = classify_energy_price(current_price, low_to_medium_threshold, medium_to_high_threshold)
    
    # Get the filling limit factor based on the energy price state (NOT USED CURRENTLY!)
    filling_limit_factor = filling_thresholds(energy_price_state)  

    density_at_filling_pressure = ((HP_tank_capacity * filling_limit_factor) + cushion_mass_per_tank) / volume_per_tank
    filling_pressure = CP.PropsSI('P', 'D', density_at_filling_pressure, 'T', T, 'Hydrogen')  # Pa (filling pressure based on filling limit factor)

    # Get the emptying limit factor based on the energy price state (NOT USED CURRENTLY!)
    emptying_limit_factor = emptying_thresholds(energy_price_state)

    # Hydrogen production at time step i
    hydrogen_production_potential = calculate_hydrogen_production(wind_power, number_of_wind_turbines, i, max_power_output, molar_mass_hydrogen)

    # Adding the production to the cumulative hydrogen production
    cumulative_hydrogen_production_potential += hydrogen_production_potential                          

    # Initial values
    startup_unhandled_incoming = 0 
    unhandled_incoming = 0  

    # Check if it is still startup phase
    if startup:
        startup, startup_unhandled_incoming = hp_tanks.new_startup_process(tanks_filling_during_startup, simultaneously_filling_tanks, HP_tank_capacity, hydrogen_production_potential, startup)
        overall_hydrogen_demand = 0
        
    # Otherwise, proceed with the normal process
    else:
        overall_hydrogen_demand, total_reserved_hydrogen, unhandled_incoming, total_unhandled_demand, total_energy_required_outgoing, total_heat_generated_outgoing = hp_tanks.new_main_process(HP_tank_capacity, emptying_limit_factor, truck_capacity, total_reserved_hydrogen, weekend, overall_hydrogen_demand, simultaneously_emptying_tanks, 
                                                                                                         filling_limit_factor, number_of_filling_stations, filling_stations, min_flowrate, max_flowrate, simultaneously_filling_tanks, 
                                                                                                         total_hydrogen_in_storage, total_capacity_of_storage, hydrogen_production_potential, time_until_weekend, current_time, total_unhandled_demand,
                                                                                                         cushion_mass_per_tank, volume_per_tank, T, isentropic_efficiency, mechanical_efficiency, COP_cooling,
                                                                                                         out_h_200bar_10C, out_h_200_to_500bar_stage_1_out_s, out_h_200_to_500bar_stage_1_cooled, out_h_500bar_stage_2_out_s, out_h_500bar_stage_2_cooled,
                                                                                                         cooldown_time)
        

    # Total hydrogen demand    
    cumulative_hydrogen_demand += overall_hydrogen_demand

    # Total heat generated from the compressors (MJ)
    cumulative_heat_generated += (total_heat_generated_outgoing + heat_generated_filling) / 1e6  

    # Total energy required for outgoing hydrogen
    overall_total_energy_required_outgoing += total_energy_required_outgoing  

    # Energy cost in euros (Convert MJ to kWh by dividing by 3.6)
    energy_cost_outgoing = (total_energy_required_outgoing / 3.6) * current_price  
    total_energy_cost_outgoing += energy_cost_outgoing  


    # Total unhandled hydrogen (not able to enter the storage tanks)
    total_unhandled_hydrogen += unhandled_incoming + startup_unhandled_incoming

    actual_hydrogen_production = hydrogen_production_potential - (unhandled_incoming + startup_unhandled_incoming)
    cumulative_actual_hydrogen_production += actual_hydrogen_production



    # Calculate the energy required for filling the tanks (USE ONE OF THE TWO FOLLOWING FUNCTIONS)

    # Use this one at 100 bar (only 2 stages required)
    energy_required_filling, heat_generated_filling = energy_requirement_for_filling(production_pressure, filling_pressure, T_production, isentropic_efficiency, mechanical_efficiency, COP_cooling, actual_hydrogen_production, fill_h_40bar_10C, fill_s_40bar_10C)

    # Use this one at pressure higher than 100 bar (3 stages required)
    #energy_required_filling, heat_generated_filling = filling_compressors_3stages(production_pressure, filling_pressure, T_production, actual_hydrogen_production, isentropic_efficiency, COP_cooling, mechanical_efficiency, fill_h_40bar_10C, fill_s_40bar_10C)



    # Total energy required for compression and filling
    total_energy_required_filling += energy_required_filling 

    energy_cost_filling = (energy_required_filling / 3.6) * current_price  # Energy cost in euros (Convert MJ to kWh by dividing by 3.6)
    total_energy_cost_filling += energy_cost_filling  # Total energy cost in euros     
    
    # Total energy cost in euros
    total_energy_cost = total_energy_cost_filling + total_energy_cost_outgoing  


    # Store the results for plotting
    time_steps.append(i)
    energy_price_over_time.append(current_price)
    demand_values.append(overall_hydrogen_demand)

    #for idx, tank in enumerate(hp_tanks.tanks):
        #HP_tanks_mass_values[idx].append(tank.net_mass_in_tank)

    total_hydrogen_in_storage = sum(tank.net_mass_in_tank for tank in hp_tanks.tanks)
    HP_total_mass_values.append(total_hydrogen_in_storage)

    production_values.append(hydrogen_production_potential)
    actual_production_values.append(actual_hydrogen_production)


    


# Final mass in high pressure tanks
final_HP_mass = sum(tank.net_mass_in_tank for tank in hp_tanks.tanks) 

# Convert from kg/min to kg/h
production_kg_h = [production * 60 for production in production_values] 
demand_values_kg_h = [demand * 60 for demand in demand_values]  

# Convert time from minutes to hours and days
time_hours = [t / 60 for t in time_steps]
time_days = [t / (24 * 60) for t in time_steps]

# Number of trucks filled
total_trucks_filled = sum(station.trucks_filled for station in filling_stations)

# Mass balance error check
mass_balance_error = cumulative_hydrogen_production_potential - cumulative_hydrogen_demand - final_HP_mass - total_unhandled_hydrogen

# Convert energy values from MJ to kWh
total_energy_required_filling_kWh = total_energy_required_filling / 3.6  
total_energy_required_outgoing_kWh = overall_total_energy_required_outgoing / 3.6  


# KPI's
storage_capacity_kg = total_capacity_of_storage  # kg
storage_capacity_density = total_capacity_of_storage / HP_total_volume  # kg/m³

energy_consumption_kWh = (total_energy_required_filling_kWh + total_energy_required_outgoing_kWh)
energy_consumption_MWh = energy_consumption_kWh / 1000  # MWh
energy_consumption_kWh_per_kg = energy_consumption_kWh / (cumulative_hydrogen_demand + final_HP_mass)

#heat generated
heat_generated_total = cumulative_heat_generated  # MJ
heat_generated_per_kg = cumulative_heat_generated / (cumulative_hydrogen_demand + final_HP_mass)  # MJ/kg
heat_generated_per_MWh = cumulative_heat_generated / (energy_consumption_MWh)  # MJ/MWh


# LCHS (OPEX known, CAPEX required) (For now only energy costs)
OPEX = total_energy_cost
OPEX_per_year = OPEX / (simulation_hours / (24 * 365))  # €/year
OPEX_per_kg = total_energy_cost / cumulative_hydrogen_demand  # €/kg



# Delivery Delay
total_delay = sum(station.total_delay for station in filling_stations)  # Total delay in minutes
average_delay_per_truck = total_delay / total_trucks_filled if total_trucks_filled > 0 else 0  # Average delay per truck in minutes


realized_production = ((cumulative_hydrogen_production_potential - total_unhandled_hydrogen) / cumulative_hydrogen_production_potential) * 100  # %

# From delivery --> Income,  

# Total cost of everything (Total plant)
price_of_hydrogen_per_kg = 8 # €/kg (price of hydrogen per kg delivered to the customer)
total_income = cumulative_hydrogen_demand * price_of_hydrogen_per_kg  # € (income from selling hydrogen)
yearly_income = total_income / (simulation_hours / (24 * 365))  # € (yearly income from selling hydrogen)



# Printing Results 
print("Final Results:")
print(f"    Simulation time: {(simulation_hours / 24)} days")
print(f"    Total Trucks Filled: {total_trucks_filled}")
print()
print(f"    Total Hydrogen Outgoing: {cumulative_hydrogen_demand:.2f} kg")
print(f"    Total Unhandled Incoming: {total_unhandled_hydrogen:.2f} kg")
print(f"    Final Mass in High Pressure Tanks: {final_HP_mass:.2f} kg")
print()
print(f"    Total Energy Required: {total_energy_required_filling_kWh + total_energy_required_outgoing_kWh:.2f} kWh")
print()
print("     Key Performance Indicators (KPI's):")
print(f"    Storage Capacity: {storage_capacity_kg:.2f} kg")
print(f"    Storage Capacity: {storage_capacity_density:.2f} kg/m³")
print()
print(f"    Energy Consumption: {energy_consumption_MWh:.2f} MWh")
print(f"    Energy Consumption: {energy_consumption_kWh_per_kg:.2f} kWh/kg")
print()
print(f"    Heat Generated: {cumulative_heat_generated:.2f} MJ")
print(f"    Heat Generated: {heat_generated_per_kg:.2f} MJ/kg")
print()
print(f"    OPEX (energy costs): € {OPEX:.2f}")
print(f"    OPEX per year: € {OPEX_per_year:.2f} €/year")
print(f"    Energy costs per kg of hydrogen delivered: € {OPEX_per_kg:.2f} €/kg")
print()
print(f"    Average Delay per Truck: {average_delay_per_truck:.2f} minutes")
print(f"    Total Delay: {total_delay:.2f} minutes")
print()
print(f"    Hydrogen Production Potential: {cumulative_hydrogen_production_potential:.2f} kg")
print(f"    Hydrogen Production Actual: {cumulative_actual_hydrogen_production:.2f} kg")
print(f"    Realized Production: {realized_production:.2f} %")
print()
print(f"    Total Income from selling hydrogen: € {total_income:.2f}") 
print(f"    Yearly Income from selling hydrogen: € {yearly_income:.2f}")



# ==========================================================
# 1️⃣ Set global style 
# ==========================================================
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'grid.alpha': 0.6,
    'lines.linewidth': 2.5
})


# ==========================================================
# 2️⃣ Plot: Total Mass in HP Tanks
# ==========================================================
plt.figure()
plt.plot(time_days, HP_total_mass_values, label='Total Mass in HP Tanks (kg)', color='green')
plt.axhline(y=total_capacity_of_storage, color='red', linestyle='--', label='Max Capacity')
plt.xlabel('Time (days)')
plt.ylabel('Total Mass in HP Tanks (kg)')
plt.title('Total Mass in High Pressure Tanks Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()


# ==========================================================
# 3️⃣ Plot: Hydrogen Demand Over Time
# ==========================================================
plt.figure()
plt.plot(time_days, demand_values_kg_h, label='Hydrogen Demand (kg/h)', color='purple')
plt.xlabel('Time (days)')
plt.ylabel('Hydrogen Demand (kg/h)')
plt.title('Hydrogen Demand Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()


# ==========================================================
# 4️⃣ Plot: Hydrogen Production Over Time
# ==========================================================
plt.figure()
plt.plot(time_days, production_kg_h, label='Hydrogen Production (kg/h)', color='orange')
plt.xlabel('Time (days)')
plt.ylabel('Hydrogen Production (kg/h)')
plt.title('Hydrogen Production Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()


# ==========================================================
# 5️⃣ Plot: Energy Price Trend
# ==========================================================
plt.figure()
plt.plot(time_days, energy_price_over_time, label="Energy Price (€/kWh)", color="blue")
#plt.axhline(y=low_to_medium_threshold, color="green", linestyle="--", label="Low Price Cutoff")
#plt.axhline(y=medium_to_high_threshold, color="orange", linestyle="--", label="Medium Price Cutoff")
plt.xlabel("Time (days)")
plt.ylabel("Energy Price (€/kWh)")
plt.title("Energy Price Trend Over Simulation")
plt.legend()
plt.grid(True)
plt.tight_layout()


# ==========================================================
# 6️⃣ Show all plots
# ==========================================================
plt.show()
