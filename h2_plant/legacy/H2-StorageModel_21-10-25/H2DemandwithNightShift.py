import random

random.seed(42)  # For reproducibility

def generate_random_truck_level():
    if random.random() < 0.2:  # 20% chance
        return random.uniform(0.0, 0.3)
    else:                      # 70% chance
        return 0.0
    
truck_level_factor = generate_random_truck_level()

class Dockingstation:
    def __init__(self, cooldown_time):
        self.truck_level = 0
        self.total_cooldown = cooldown_time #random.randint(150, 180)  # Cooldown time in minutes
        self.cooldown_time = self.total_cooldown
        self.reserved = False
        self.reserved_hydrogen = 0
        self.completed = False
        self.locked_time_until_weekend = None
        self.filling = False
        self.calculated_flowrate = 0
        self.hydrogen_demand = 0
        self.delay_counter = 0  # Counter for how many times the truck couldn't reserve hydrogen

    # Process of the active docking station
    def active_dockingstation(self, truck_capacity, net_available_hydrogen, max_h2_per_station, time_until_weekend, max_flowrate, min_flowrate, inactive_dock_cooldown_time, current_time, cooldown_time):

        # Initialize completed value (Just to be sure)
        self.completed = False

        # Initialize delay counter
        self.delay_counter = 0

        # Check if the night shift is active
        night_shift_active = get_night_shift_status(current_time)

        # Calculate the time until the night shift starts
        time_until_night_shift = time_until_night_shift_calculation(current_time)

        # If the hydrogen to fill the truck is not yet reserved, check if it can be reserved
        # This is done so that the truck can fill up in one go
        if not self.reserved:

            # The minimum time needed to fill the truck
            min_time_needed_to_fill = (truck_capacity - self.truck_level) / max_flowrate

            # Check if there is time to fill a truck before the weekend starts
            if time_until_weekend < min_time_needed_to_fill + 0.01:

                # Not enough time to fill the truck before the weekend
                self.reserved_hydrogen = 0
                self.hydrogen_demand = 0

                return self.completed, self.hydrogen_demand, self.delay_counter
            
            if night_shift_active:
                # This means that the night shift is active, and that the truck failed to reserve hydrogen before the night shift started.
                # This means that the truck will not be able to fill during the night shift, and therefore needs to wait until the next day to reserve and fill.
                # Waiting with reserving hydrogen should not be a problem, as the storage should fill up during the night shift, and the truck can reserve hydrogen directly after the night shift ends.


                self.reserved_hydrogen = 0
                self.hydrogen_demand = 0
                return self.completed, self.hydrogen_demand, self.delay_counter
            
            # Check if there is enough hydrogen available to fill the truck
            # If so, that amount of hydrogen is reserved for the truck
            if net_available_hydrogen >= (truck_capacity - self.truck_level):
                self.reserved_hydrogen = truck_capacity - self.truck_level
                net_available_hydrogen -= self.reserved_hydrogen
                self.reserved = True

            else:
                self.delay_counter = 1
                self.reserved_hydrogen = 0
                self.hydrogen_demand = 0

        # If the hydrogen for the truck has been reserved, calculate the flowrate
        if self.reserved and not self.filling:
             
             self.calculated_flowrate = calculate_flowrate(max_flowrate, min_flowrate, inactive_dock_cooldown_time, self.truck_level, truck_capacity, time_until_weekend, time_until_night_shift)

        if self.calculated_flowrate <= 0:
                self.filling = False
                    
        else:
            self.filling = True


        # This is the filling process of the truck (Once the hydrogen is reserved and the flowrate is calculated)
        if self.reserved and self.filling:

            # The hydrogen demand for this truck, this iteration, is the calculated flowrate or the remaining capacity of the truck, whichever is lower. (max hydrogen per station is also there, but should be reconsidered)
            self.hydrogen_demand = min (truck_capacity - self.truck_level, self.calculated_flowrate, max_h2_per_station)
            self.truck_level += self.hydrogen_demand
            self.reserved_hydrogen -= self.hydrogen_demand

            # The time remaining to fill the truck is the flowrate divided by the remaining capacity of the truck
            self.time_remaining_to_fill = (truck_capacity - self.truck_level) / self.calculated_flowrate

            # If the truck is filled, set it to completed and reset the truck level, cooldown time, reserved hydrogen, reserved status, and filling status
            if self.truck_level >= truck_capacity:
                self.completed = True
                self.total_cooldown = cooldown_time #random.randint(150, 180)  # Reset cooldown time in minutes
                self.cooldown_time = self.total_cooldown
                self.truck_level = 0  # generate_random_truck_level() * truck_capacity
                self.reserved = False
                self.reserved_hydrogen = 0
                self.filling = False

        return self.completed, self.hydrogen_demand, self.delay_counter
    


    # Process of the inactive docking station
    def inactive_dockingstation(self, net_available_hydrogen, truck_capacity, active_dock_time_remaining_to_fill, active_filling, max_flowrate, time_until_weekend):

        # Check if the dock is still cooling down (no truck arrived yet)
        if self.cooldown_time > 0:
            self.cooldown_time -= 1

            if self.cooldown_time <= 0:
                self.cooldown_time = 0

            else:
                return 0
            

        min_projected_fill_time = (truck_capacity - self.truck_level) / max_flowrate 

        # If not reserved yet, check if the hydrogen for the truck can already be reserved, so it can directly start filling when it becomes active
        if not self.reserved:

            # First check if the active dock is filling a truck, if not, reservation for the inactive dock is not a priority
            if active_filling:
                 
                 # Check if there is enough time to fill the truck in the active dock and then this one before the weekend starts
                 if active_dock_time_remaining_to_fill + min_projected_fill_time > time_until_weekend:
                    # Not enough time to fill the truck before the weekend
                    self.reserved_hydrogen = 0
                    self.reserved = False
                    return 0
                 
                 else:
                    # Check if hydrogen can already be reserved for the truck in the inactive dock
                    if net_available_hydrogen >= (truck_capacity - self.truck_level):
                        self.reserved_hydrogen = truck_capacity - self.truck_level
                        net_available_hydrogen -= self.reserved_hydrogen
                        self.reserved = True

                    else:
                        self.reserved_hydrogen = 0
                        self.reserved = False
            
                      
            
        if self.reserved:

            self.reserved_hydrogen = truck_capacity - self.truck_level




# This function calculates the flowrate for filling a truck based on various conditions
# It takes into account the maximum and minimum flowrate, cooldown time, truck level, truck capacity, time until weekend, and time until night shift
# It returns the calculated flowrate for filling the truck
def calculate_flowrate (max_flowrate, min_flowrate, cooldown_time, truck_level, truck_capacity, time_until_weekend, time_until_night_shift, night_shift_duration = 480):
    
    # Night shift = 8 hours
    # Max time for filling = 10 hours

    # There is a window that start 2 hours before the night shift starts and ends 30 minutes before the night shift starts
    # During this window, a truck that will be filled during the night shift should start filling
    begin_window = 60 # 120
    end_window = 15 # 30
    until_begin_night_truck_window = time_until_night_shift - begin_window
    until_end_night_truck_window = time_until_night_shift - end_window

    minimum_time_to_fill = (truck_capacity / max_flowrate) 


    # The time needed to fill another truck after this one, before the night shift starts
    # It also includes another buffer time to help the system and get rid of some cases where the times were on the edge. (now deleted, but might be added again if needed)
    # This solved some cases where the flowrate was too high (higher than the max flowrate).
    fill_time_for_next_truck = cooldown_time + minimum_time_to_fill

    
    # If it is already past the start of the before night shift window, then the truck should be filled during the night shift
    # The flowrate will be such that the truck is finished filling when the night shift ends
    if time_until_night_shift <= 60: # 120
        calculated_flowrate = (truck_capacity - truck_level) / (time_until_night_shift + night_shift_duration)
        return calculated_flowrate
    
    # --- New check: Prevent trucks from starting if they would finish during the night shift ---
    # If the time required to fill this truck exceeds the remaining time before night shift (minus the buffer window),
    # then filling would overlap into the night shift → do not start.
    if minimum_time_to_fill > (time_until_night_shift - end_window):
        return 0
    # If a truck can be filled, such that it finishes in the before night shift window, then calculate the flowrate
    elif until_begin_night_truck_window <= minimum_time_to_fill < until_end_night_truck_window:
         
        if minimum_time_to_fill > until_end_night_truck_window:
        # Not enough time to fill at max flowrate, don't start
            return 0
         
         # --- Block overlapping fills before night shift ---
        #if minimum_time_to_fill + cooldown_time > time_until_night_shift:
            #return 0
         
        else:
            # Safe to fill, set flowrate so truck finishes in window (but never above max)
            calculated_flowrate = (truck_capacity - truck_level) / until_end_night_truck_window
            calculated_flowrate = min(calculated_flowrate, max_flowrate)
            calculated_flowrate = max(calculated_flowrate, min_flowrate)
            return calculated_flowrate
    
    # If there is not enough time to fill another truck after this one, before the night shift starts, then this truck should finish filling up in the before night shift window.
    elif time_until_night_shift - fill_time_for_next_truck < end_window:
        #This makes sure that the truck is filled up during the before night shift window. 

        calculated_flowrate = (truck_capacity - truck_level) / until_end_night_truck_window 

        if calculated_flowrate > max_flowrate:
            
            calculated_flowrate = 0
             #Do not fill the truck

        return calculated_flowrate

    # Otherwise, under normal conditions, the truck can be filled based on the cooldown time of the inactive dock and the time until the weekend
    else:

        if cooldown_time > 0:
            calculated_flowrate = (truck_capacity - truck_level) / cooldown_time

            required_flowrate = (truck_capacity - truck_level) / time_until_weekend

        else:
            calculated_flowrate = max_flowrate
            return calculated_flowrate

        # Not possible to go beyond the maximum flowrate
        if required_flowrate > max_flowrate:
            print("not possible")

            calculated_flowrate = 0
            # Do not fill the truck
        
        if calculated_flowrate < required_flowrate:
            calculated_flowrate = required_flowrate

        if calculated_flowrate <= min_flowrate:
                calculated_flowrate = min_flowrate

        elif calculated_flowrate >= max_flowrate:
                calculated_flowrate = max_flowrate

        return calculated_flowrate





class FillingStations:
    def __init__(self, station_id, cooldown_time):
        self.station_id = station_id
        self.docks = [Dockingstation(cooldown_time), Dockingstation(cooldown_time)]
        self.active_dock = 0
        self.inactive_dock = 1
        self.total_hydrogen_demand = 0
        self.trucks_filled = 0
        self.total_delay = 0



    # This function is the main process of the filling stations
    def dockstation_process(self, min_flowrate, max_flowrate, truck_capacity, net_available_hydrogen, max_h2_per_station, weekend, time_until_weekend, current_time, cooldown_time):

        # If it is weekend, no trucks can be filled
        if weekend:
            return 0

        # Set the active and inactive docks
        # The active dock is the one that is currently filling a truck, the inactive dock is the one that is waiting for the active dock to finish filling
        active = self.docks[self.active_dock]
        inactive = self.docks[self.inactive_dock]

        # The cooldown time of the inactive dock
        inactive_dock_cooldown_time = max(inactive.cooldown_time - 1, 0) # minus 1 because this value comes from the previous iteration

        
        # Run the active docking station process
        active.completed, active.hydrogen_demand, active.delay_counter = active.active_dockingstation(truck_capacity, net_available_hydrogen, max_h2_per_station, time_until_weekend, max_flowrate, min_flowrate, inactive_dock_cooldown_time, current_time, cooldown_time)

        # The hydrogen demand of the filling station is the hydrogen demand of the active dock
        self.total_hydrogen_demand = active.hydrogen_demand

        self.total_delay += active.delay_counter

        # If the active dock has completed filling a truck, switch the active and inactive docks
        if active.completed:
            self.trucks_filled += 1
            active.completed = False

            self.active_dock, self.inactive_dock = self.inactive_dock, self.active_dock


            active = self.docks[self.active_dock]
            inactive = self.docks[self.inactive_dock]


        # Reset the active and inactive docks
        active = self.docks[self.active_dock]
        inactive = self.docks[self.inactive_dock]

        if active.filling:
            active_dock_time_remaining_to_fill = active.time_remaining_to_fill
            active_filling = True

        if not active.filling:
            active_filling = False
            active_dock_time_remaining_to_fill = 0

        # Run the inactive docking station process
        inactive.inactive_dockingstation(net_available_hydrogen, truck_capacity, active_dock_time_remaining_to_fill, active_filling, max_flowrate, time_until_weekend)


        return self.total_hydrogen_demand
    


def get_night_shift_status(current_time):
    # Night shift windows in minutes since start of week (Monday–Thursday only)
    night_shifts = [
        (1380, 1860),   # Mon 23:00 – Tue 07:00
        (2820, 3300),   # Tue 23:00 – Wed 07:00
        (4260, 4740),   # Wed 23:00 – Thu 07:00
        (5700, 6180),   # Thu 23:00 – Fri 07:00
        (7140, 7620),    # Fri 23:00 – Sat 07:00 (if weekend operation is considered)
        #(8580, 9060),    # Sat 23:00 – Sun 07:00 (if weekend operation is considered)
        #(10020, 420)    # Sun 23:00 – Mon 07:00 (if weekend operation is considered)
    ]

    time_in_week = current_time % 10080  # Wraps for multi-week simulations

    night_shift_active = any(start <= time_in_week < end for start, end in night_shifts)

    return night_shift_active


def time_until_night_shift_calculation(current_time):

    night_shift_starts = [1380, 2820, 4260, 5700, 7140]  # Start times of night shifts in minutes since start of week (Monday–Friday) (weekend of 1 day is considered)

    time_in_week = current_time % 10080  # Wraps for multi-week simulations

    for start_time in night_shift_starts:
        if time_in_week < start_time:
            time_until_night_shift = start_time - time_in_week
            return time_until_night_shift

    # No night shifts left in current week — return time until next week's first shift
    time_until_night_shift = (night_shift_starts[0] + 10080) - time_in_week
    return time_until_night_shift


