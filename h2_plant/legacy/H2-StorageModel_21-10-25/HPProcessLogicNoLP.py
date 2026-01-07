import random
import CoolProp.CoolProp as CP

from Outgoingcompressor import energy_required_for_outgoing_1_2stages, energy_required_for_outgoing_1_3stages, energy_required_for_outgoing_2

random.seed(42)  # For reproducibility

# Making a class of the individual tanks
class Tank:

    def __init__ (self):
        self.net_mass_in_tank = 0
        self.outgoing_mass = 0
        self.pressure = 0
        self.state = "Idle"

# The class with all the tanks and control logic
# This class manages the tanks, their states, and the filling/emptying processes
# It also handles the startup process and the filling of tanks during normal operation

class HP_Tanks:
    # Initialize the HP_Tanks class with a specified number of tanks
    def __init__(self, number_of_tanks):
        self.tanks = [Tank() for _ in range(number_of_tanks)]  

    # Startup process for filling tanks during startup
    # This process fills a specified number of tanks during startup
    def new_startup_process(self, tanks_filling_during_startup, simultaneously_filling_tanks, HP_tank_capacity, hydrogen_production_potential, startup):

        startup_idle_tanks = [tank for tank in self.tanks[:tanks_filling_during_startup] if tank.state == "Idle"]
        startup_filling_tanks = [tank for tank in self.tanks[:tanks_filling_during_startup] if tank.state == "Filling"]

        # Check if the maximum number of tanks are filling
        if len(startup_filling_tanks) < simultaneously_filling_tanks:

            # If not, sort the idle tanks based on their mass (ascending order) 
            sorted_startup_idle_tanks = sorted(startup_idle_tanks, key=lambda t: t.net_mass_in_tank)

            # Now check the sorted idle tanks if they can be changed to filling state
            for tank in sorted_startup_idle_tanks:
                if tank.net_mass_in_tank < HP_tank_capacity:
                    tank.state = "Filling"
                    startup_filling_tanks = [tank for tank in self.tanks[:tanks_filling_during_startup] if tank.state == "Filling"]

                    # If the maximum number of filling tanks is reached, break the loop
                    if len(startup_filling_tanks) >= simultaneously_filling_tanks:
                        break


        startup_filling_tanks = [tank for tank in self.tanks[:tanks_filling_during_startup] if tank.state == "Filling"]

        # The hydrogen to add to the filling tanks is the hydrogen produced
        startup_hydrogen_to_add = hydrogen_production_potential
        remaining_startup_filling_tanks = len(startup_filling_tanks)

        # Distribute the hydrogen evenly among the filling tanks during startup
        for tank in startup_filling_tanks:

            hydrogen_add_to_tank = min(startup_hydrogen_to_add / remaining_startup_filling_tanks, HP_tank_capacity - tank.net_mass_in_tank)     # any maximum flowrate is left out
            tank.net_mass_in_tank += hydrogen_add_to_tank
            startup_hydrogen_to_add -= hydrogen_add_to_tank
            remaining_startup_filling_tanks -= 1

            # If the tank is full, change its state to idle
            if tank.net_mass_in_tank >= HP_tank_capacity:
                tank.state = "Idle"

        # If there is still hydrogen to add, but there is not enough capacity, it is considered unhandled
        if startup_hydrogen_to_add > 0:
            startup_unhandled_incoming = startup_hydrogen_to_add

        else:
            startup_unhandled_incoming = 0

        # Check if the number of tanks that are filled is equal to the number of tanks that should be filled during startup
        # If so, the startup process can end
        if all(tank.net_mass_in_tank >= HP_tank_capacity for tank in self.tanks[:tanks_filling_during_startup]):
            print("DEBUG: Startup should end NOW!")  # ✅ Check if this prints
            startup = False

        return startup, startup_unhandled_incoming



    
    # This function fills the tanks that are in filling state
    def new_filling_tanks(self, hydrogen_production_potential, HP_tank_capacity, filling_limit_factor):

        filling_tanks = [tank for tank in self.tanks if tank.state == "Filling"]
        filling_tanks.sort(key=lambda t: t.net_mass_in_tank, reverse=True)

        total_hydrogen_to_add = hydrogen_production_potential

        remaining_filling_tanks = len(filling_tanks)

        # The filling tanks are sorted based on their mass (POINT OF DISCUSSION: SORT THEM DESCENDING OR ASCENDING?)
        # Hydrogen is spread evenly over the filling tanks, but only up to their limit
        for tank in filling_tanks:
            hydrogen_to_add_to_tank = min(total_hydrogen_to_add / remaining_filling_tanks, (HP_tank_capacity * filling_limit_factor) - tank.net_mass_in_tank )
            tank.net_mass_in_tank += hydrogen_to_add_to_tank
            total_hydrogen_to_add -= hydrogen_to_add_to_tank
            remaining_filling_tanks -= 1

        # If there is still hydrogen to add, it means that there is not enough capacity in the filling tanks
        # All the hydrogen that could not be added is considered unhandled
        if total_hydrogen_to_add > 0:
            unhandled_incoming = total_hydrogen_to_add
        else:
            unhandled_incoming = 0

        return unhandled_incoming
    


    # This function handles the emptying of tanks
    def emptying_tanks(self, overall_hydrogen_demand, HP_tank_capacity, emptying_limit_factor, total_unhandled_demand, cushion_mass_per_tank, volume_per_tank, T, isentropic_efficiency, mechanical_efficiency, COP_cooling,
                       out_h_200bar_10C, out_h_200_to_500bar_stage_1_out_s, out_h_200_to_500bar_stage_1_cooled, out_h_500bar_stage_2_out_s, out_h_500bar_stage_2_cooled):

        # Sort the emptying tanks based on their mass (descending order) (POINT OF DISCUSSION: SORT THEM DESCENDING OR ASCENDING?)
        emptying_tanks = [tank for tank in self.tanks if tank.state == "Emptying"]
        emptying_tanks.sort(key=lambda t: t.net_mass_in_tank)

        remaining_emptying_tanks = len(emptying_tanks)

        remaining_demand = overall_hydrogen_demand

        # Run all the emptying tanks
        for tank in emptying_tanks:
            tank.outgoing_mass = 0

            # Check the pressure of the tank for later (compressor calculations)
            density_at_pressure = (tank.net_mass_in_tank + cushion_mass_per_tank) / volume_per_tank
            tank.pressure = CP.PropsSI('P', 'D', density_at_pressure, 'T', T, 'Hydrogen')               # Maybe unnecessary to calculate every itteration, can also be done just at the end of the simulation just once
            
            # Empty the tank based on the remaining demand and capacity
            hydrogen_to_empty = max(min(remaining_demand / remaining_emptying_tanks, tank.net_mass_in_tank - (HP_tank_capacity * emptying_limit_factor)), 0)    # For now, emptying flowrate is left out, but should be considered in the future
            tank.net_mass_in_tank -= hydrogen_to_empty
            remaining_demand -= hydrogen_to_empty
            remaining_emptying_tanks -= 1
            tank.outgoing_mass = hydrogen_to_empty



        # At some points, there is demand remaining, caused by the emptying limit. This results in demand not being fulfilled by the emptying tanks.
        # As a safeguard, the remaining demand will be distributed evenly over the emptying tanks, for now removing the emptying limit. 
        # Exactly the root cause of the problem is not known, only that it is caused by the emptying limit factor.
        # The amount that this happens is still tracked, and added up to the total unhandled demand (still only a small amount compared to the total mass). 

        if remaining_demand > 0:
            for tank in emptying_tanks:
                extra_removed_hydrogen = remaining_demand / len(emptying_tanks)
                tank.net_mass_in_tank -= extra_removed_hydrogen
                tank.outgoing_mass += extra_removed_hydrogen
                
            
            total_unhandled_demand += remaining_demand


        # Now with the demand per emptying tank, we need to calculate the compressor energy requirements for each tank.

        total_energy_required_outgoing_1 = 0 # Initialize the total energy required for the first stage of outgoing compression

        # Calculate the energy required for each emptying tank, to compress the outgoing hydrogen to 200 bar
        # Here again, there is a choice between a 2 stage and 3 stage compressor
        # If the minimum pressure is 40 bar, then a 3 stage compressor is needed to keep below 85 degrees C
        # If the minimum pressure is 70 bar, then a 2 stage compressor is sufficient to keep below 85 degrees C

        # CHOOSE ONE OF THE TWO BELOW:
        for tank in emptying_tanks:
            energy_required_outgoing_per_tank, heat_generated_outgoing_per_tank_1 = energy_required_for_outgoing_1_2stages(tank.pressure, isentropic_efficiency, mechanical_efficiency, COP_cooling, tank.outgoing_mass)
            #energy_required_outgoing_per_tank, heat_generated_outgoing_per_tank_1 = energy_required_for_outgoing_1_3stages(tank.pressure, isentropic_efficiency, mechanical_efficiency, COP_cooling, tank.outgoing_mass)
            total_energy_required_outgoing_1 += energy_required_outgoing_per_tank

        total_outgoing_mass = sum(tank.outgoing_mass for tank in emptying_tanks)

        # Now calculate the energy required to compress the combined outgoing hydrogen from 200 bar to 500 bar
        energy_required_outgoing_2, heat_generated_outgoing_per_tank_2 = energy_required_for_outgoing_2(isentropic_efficiency, mechanical_efficiency, COP_cooling, 
                                                                    total_outgoing_mass, out_h_200bar_10C, out_h_200_to_500bar_stage_1_out_s, out_h_200_to_500bar_stage_1_cooled, out_h_500bar_stage_2_out_s, out_h_500bar_stage_2_cooled)

        total_energy_required_outgoing = (total_energy_required_outgoing_1 + energy_required_outgoing_2) 

        total_heat_generated_outgoing = heat_generated_outgoing_per_tank_1 + heat_generated_outgoing_per_tank_2

        return total_unhandled_demand, total_energy_required_outgoing, total_heat_generated_outgoing    





    # Updating the tanks that are in emptying state
    # This function checks if the tanks that are in emptying state can remain in that state or should change to idle
    def update_state_emptying_tanks(self, HP_tank_capacity, emptying_limit_factor, truck_capacity, total_reserved_hydrogen, weekend, overall_hydrogen_demand, total_hydrogen_in_storage, total_capacity_of_storage):

        emptying_tanks = [tank for tank in self.tanks if tank.state == "Emptying"]

        total_emptying_tanks_mass = sum(tank.net_mass_in_tank for tank in emptying_tanks)

        state_changed = False
        
        for tank in emptying_tanks:
            if tank.net_mass_in_tank <= (HP_tank_capacity * emptying_limit_factor) or (total_emptying_tanks_mass < truck_capacity and total_reserved_hydrogen <= 0) or weekend == True and overall_hydrogen_demand <= 0:
                tank.state = "Idle"
                state_changed = True

            
        if not state_changed:
            # If there has been no change in the state of the emptying tanks, there will be a check if a tank can be replaced by a tank that has significantly more mass in it
            if total_hydrogen_in_storage > 0 * total_capacity_of_storage:   # Changed to total hydrogen in storage (instead of LP net mass) (This was a prerequisite for the changing of tanks, but is probably not needed anymore)

                emptying_tanks = [tank for tank in self.tanks if tank.state == "Emptying"]
                filling_tanks = [tank for tank in self.tanks if tank.state == "Filling"]
                idle_tanks = [tank for tank in self.tanks if tank.state == "Idle"]
                potential_emptying_tanks = filling_tanks + idle_tanks

                # Sort the tanks based on their mass (potential tanks in descending order, emptying tanks in ascending order)
                sorted_potential_tanks = sorted(potential_emptying_tanks, key=lambda t: t.net_mass_in_tank, reverse=True)
                sorted_emptying_tanks = sorted(emptying_tanks, key=lambda t: t.net_mass_in_tank)

                # Now take the potential tank with the most mass and the emptying tank with the least mass
                if sorted_potential_tanks and sorted_emptying_tanks:
                    to_replace_tank = sorted_emptying_tanks[0]
                    replacement_tank = sorted_potential_tanks[0]

                    # If there is a potential tank that has more than 110% of the mass of the tank that is being replaced, they switch states
                    if to_replace_tank.net_mass_in_tank * 1.1 < replacement_tank.net_mass_in_tank:
                        to_replace_tank.state = "Idle"
                        replacement_tank.state = "Emptying"

        


    # Updating the tanks that are in filling or idle state
    # This function checks if the tanks that are in filling or idle state can remain in that state or should change to emptying
    # It also checks if the tanks that are in filling state can change to idle state
    def update_state_filling_and_idle_tanks(self, simultaneously_emptying_tanks, HP_tank_capacity, emptying_limit_factor, filling_limit_factor, simultaneously_filling_tanks):

        emptying_tanks = [tank for tank in self.tanks if tank.state == "Emptying"]
        filling_tanks = [tank for tank in self.tanks if tank.state == "Filling"]
        idle_tanks = [tank for tank in self.tanks if tank.state == "Idle"]
        potential_emptying_tanks = filling_tanks + idle_tanks
        
        number_of_emptying_tanks = len(emptying_tanks)

        # Check if there are already the maximum number of emptying tanks
        if number_of_emptying_tanks < simultaneously_emptying_tanks:

            # If not, sort the potential tanks baased on their mass (descending order)
            sorted_potential_tanks = sorted(potential_emptying_tanks, key=lambda t: t.net_mass_in_tank, reverse=True)

            # Now check the sorted potential tanks if they can be changed to emptying state
            for tank in sorted_potential_tanks:
                if tank.net_mass_in_tank > (HP_tank_capacity * emptying_limit_factor):
                    tank.state = "Emptying"
                    number_of_emptying_tanks += 1

                    # If the maximum number of emptying tanks is reached, break the loop
                    if number_of_emptying_tanks >= simultaneously_emptying_tanks:
                        break

        
        filling_tanks = [tank for tank in self.tanks if tank.state == "Filling"]

        # If there are filling tanks that are full, change them to idle state
        for tank in filling_tanks:
            if tank.net_mass_in_tank >= (HP_tank_capacity * filling_limit_factor):
                tank.state = "Idle"


        filling_tanks = [tank for tank in self.tanks if tank.state == "Filling"]
        idle_tanks = [tank for tank in self.tanks if tank.state == "Idle"]
        sorted_idle_tanks = sorted(idle_tanks, key=lambda t: t.net_mass_in_tank)

        number_of_filling_tanks = len(filling_tanks)

        # Check if there are already the maximum number of filling tanks
        if number_of_filling_tanks < simultaneously_filling_tanks:

            # If not, sort the idle tanks based on their mass (ascending order)
            sorted_idle_tanks = sorted(idle_tanks, key=lambda t: t.net_mass_in_tank)

            # Now check the sorted idle tanks if they can be changed to filling state
            for tank in sorted_idle_tanks:
                if tank.net_mass_in_tank < (HP_tank_capacity * filling_limit_factor):
                    tank.state = "Filling"
                    number_of_filling_tanks += 1

                    # If the maximum number of filling tanks is reached, break the loop
                    if number_of_filling_tanks >= simultaneously_filling_tanks:
                        break


    # This function determines the outgoing hydrogen (demand) based on the current state of the tanks and the filling stations                     
    def determine_demand(self, HP_tank_capacity, emptying_limit_factor, number_of_filling_stations, filling_stations, min_flowrate, max_flowrate, truck_capacity, weekend, time_until_weekend, current_time, cooldown_time):

        emptying_tanks = [tank for tank in self.tanks if tank.state == "Emptying"]

        #available_hydrogen = sum(tank.net_mass_in_tank - (HP_tank_capacity * emptying_limit_factor) for tank in emptying_tanks)

        # Calculates the available hydrogen in the emptying tanks
        available_hydrogen = sum(max(tank.net_mass_in_tank - (HP_tank_capacity * emptying_limit_factor), 0) for tank in emptying_tanks)

        #available_hydrogen_per_minute = sum(min(tank.net_mass_in_tank - (HP_tank_capacity * emptying_limit_factor), )

        # Calculates the maximum amount of hydrogen that ...
        max_h2_per_station = available_hydrogen / number_of_filling_stations

        # Calculates the total amount of hydrogen that is reserved by the filling stations
        total_reserved_hydrogen = sum(dock.reserved_hydrogen for fs in filling_stations for dock in fs.docks)

        # Calculates the net available hydrogen (the available hydrogen minus the reserved hydrogen)
        net_available_hydrogen = available_hydrogen - total_reserved_hydrogen

        overall_hydrogen_demand = 0 # Initialize the overall hydrogen demand

        # Run all of the filling stations
        for station in filling_stations:

            # Run the dockstation process for each filling station
            total_hydrogen_demand = station.dockstation_process(min_flowrate, max_flowrate, truck_capacity, net_available_hydrogen, max_h2_per_station, weekend, time_until_weekend, current_time, cooldown_time)

            # Update the total reserved hydrogen, net available hydrogen, and overall hydrogen demand
            total_reserved_hydrogen = sum(dock.reserved_hydrogen for dock in station.docks) ##
            net_available_hydrogen = available_hydrogen - total_reserved_hydrogen
            overall_hydrogen_demand += total_hydrogen_demand

        return overall_hydrogen_demand, total_reserved_hydrogen

        

    # This is the main process that controls the filling and emptying of tanks
    def new_main_process(self, HP_tank_capacity, emptying_limit_factor, truck_capacity, total_reserved_hydrogen, weekend, overall_hydrogen_demand, simultaneously_emptying_tanks, filling_limit_factor, 
                         number_of_filling_stations, filling_stations, min_flowrate, max_flowrate, simultaneously_filling_tanks, total_hydrogen_in_storage, total_capacity_of_storage, hydrogen_production_potential, time_until_weekend, current_time, total_unhandled_demand
                         , cushion_mass_per_tank, volume_per_tank, T, isentropic_efficiency, mechanical_efficiency, COP_cooling, out_h_200bar_10C, out_h_200_to_500bar_stage_1_out_s, out_h_200_to_500bar_stage_1_cooled, out_h_500bar_stage_2_out_s, out_h_500bar_stage_2_cooled,
                         cooldown_time):

        
            # Update the state of the tanks that are in emptying state
            self.update_state_emptying_tanks(HP_tank_capacity, emptying_limit_factor, truck_capacity, total_reserved_hydrogen, weekend, overall_hydrogen_demand, total_hydrogen_in_storage, total_capacity_of_storage)

            # Update the state of the tanks that are in filling or idle state
            self.update_state_filling_and_idle_tanks(simultaneously_emptying_tanks, HP_tank_capacity, emptying_limit_factor, filling_limit_factor, simultaneously_filling_tanks)

            # Run the tanks that are in filling state
            unhandled_incoming = self.new_filling_tanks(hydrogen_production_potential, HP_tank_capacity, filling_limit_factor)

            # Determine the outgoing hydrogen based on the current demand and what is available in the tanks
            overall_hydrogen_demand, total_reserved_hydrogen = self.determine_demand(HP_tank_capacity, emptying_limit_factor, number_of_filling_stations, filling_stations, min_flowrate, max_flowrate, truck_capacity, weekend, time_until_weekend, current_time, cooldown_time)

            # Run the tanks that are in emptying state
            total_unhandled_demand, total_energy_required_outgoing, total_heat_generated_outgoing = self.emptying_tanks(overall_hydrogen_demand, HP_tank_capacity, emptying_limit_factor, total_unhandled_demand, cushion_mass_per_tank, volume_per_tank, T, isentropic_efficiency, mechanical_efficiency, COP_cooling,
                                                                                         out_h_200bar_10C, out_h_200_to_500bar_stage_1_out_s, out_h_200_to_500bar_stage_1_cooled, out_h_500bar_stage_2_out_s, out_h_500bar_stage_2_cooled)

            return overall_hydrogen_demand, total_reserved_hydrogen, unhandled_incoming, total_unhandled_demand, total_energy_required_outgoing, total_heat_generated_outgoing




    


                        


            

        



     


        









