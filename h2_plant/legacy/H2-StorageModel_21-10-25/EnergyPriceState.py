

def classify_energy_price(current_price, low_to_medium_threshold, medium_to_high_threshold):
    """Classifies the energy price into Low, Medium, or High."""

    if current_price <= low_to_medium_threshold:
        energy_price_state = "Low"
        
    elif current_price <= medium_to_high_threshold:
        energy_price_state = "Medium"
        
    else:
        energy_price_state = "High"
    

    return energy_price_state
        

def filling_thresholds(energy_price_state):
    """Determines the filling limit factor based on the energy price state."""

    if energy_price_state == "Low":
        filling_limit_factor = 1
        
    elif energy_price_state == "Medium":
        filling_limit_factor = 1 
    
    elif energy_price_state == "High":
        filling_limit_factor = 1 
    
    return filling_limit_factor



def emptying_thresholds(energy_price_state):
    """Determines the emptying limit factor based on the energy price state."""

    if energy_price_state == "Low":
        emptying_limit_factor = 0
        
    elif energy_price_state == "Medium":
        emptying_limit_factor = 0 
    
    elif energy_price_state == "High":
        emptying_limit_factor = 0 
    
    return emptying_limit_factor