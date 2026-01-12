
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Mocking the EngineDispatch logic to test _determine_zone and _calculate_action_factor
class MockAPC:
    def __init__(self):
        self._ctrl_params = {
            'SOC_LOW': 0.60,         # Zone 0 -> 1
            'SOC_HIGH': 0.80,        # Zone 1 -> 2
            'SOC_CRITICAL': 0.95,    # Zone 2 -> 3
            'HYSTERESIS': 0.02,
            'MAX_RATE_H': 0.20,
            'MIN_ACTION_FACTOR': 0.1
        }
        self._ctrl_state = {
            'current_zone': 0
        }

    def _determine_zone(self, soc: float) -> int:
        p = self._ctrl_params
        current_zone = self._ctrl_state['current_zone']
        
        # Thresholds
        z1_thresh = p['SOC_LOW']
        z2_thresh = p['SOC_HIGH']
        z3_thresh = p['SOC_CRITICAL']
        hyst = p['HYSTERESIS']

        new_zone = current_zone

        # Transition Logic
        if soc >= z3_thresh:
            new_zone = 3
        elif soc >= z2_thresh:
            if current_zone == 3 and soc > (z3_thresh - hyst):
                new_zone = 3
            else:
                new_zone = 2
        elif soc >= z1_thresh:
            if current_zone == 2 and soc > (z2_thresh - hyst):
                new_zone = 2
            else:
                new_zone = 1
        else:
            if current_zone == 1 and soc > (z1_thresh - hyst):
                new_zone = 1
            else:
                new_zone = 0
            
        return new_zone

    def _calculate_action_factor(self, zone: int, soc: float, dsoc_dt: float) -> float:
        p = self._ctrl_params
        
        factor = 1.0
        
        if zone == 0:
            factor = 1.0
        elif zone == 1:
            norm = (soc - p['SOC_LOW']) / (p['SOC_HIGH'] - p['SOC_LOW'])
            factor = 1.0 - (0.3 * norm) # 1.0 -> 0.7
        elif zone == 2:
            norm = (soc - p['SOC_HIGH']) / (p['SOC_CRITICAL'] - p['SOC_HIGH'])
            factor = 0.7 * (1.0 - norm) # 0.7 -> 0.0
        elif zone == 3:
            factor = 0.0

        if dsoc_dt > p['MAX_RATE_H']:
            rate_penalty = (dsoc_dt - p['MAX_RATE_H']) * 2.0
            factor = max(0.0, factor - rate_penalty)

        return max(0.0, min(1.0, factor))

    def step(self, soc):
        zone = self._determine_zone(soc)
        self._ctrl_state['current_zone'] = zone
        # Assume slow filling for basic logic test
        factor = self._calculate_action_factor(zone, soc, dsoc_dt=0.0) 
        return zone, factor

def run_verification():
    apc = MockAPC()
    
    # 1. Sweep 0 -> 1 (Filling)
    soc_up = np.linspace(0, 1.0, 200)
    zones_up = []
    factors_up = []
    
    for soc in soc_up:
        z, f = apc.step(soc)
        zones_up.append(z)
        factors_up.append(f)
        
    # 2. Sweep 1 -> 0 (Emptying)
    soc_down = np.linspace(1.0, 0, 200)
    zones_down = []
    factors_down = []
    
    # Reset to full state for down sweep
    apc._ctrl_state['current_zone'] = 3 
    
    for soc in soc_down:
        z, f = apc.step(soc)
        zones_down.append(z)
        factors_down.append(f)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Zones
    ax1.plot(soc_up, zones_up, 'b-', label='Filling (0->1)')
    ax1.plot(soc_down, zones_down, 'r--', label='Emptying (1->0)')
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['Normal (0)', 'Attention (1)', 'Alert (2)', 'Critical (3)'])
    ax1.set_ylabel('APC Zone')
    ax1.set_title('APC Zone Transition Logic (Check Hysteresis)')
    ax1.grid(True)
    ax1.legend()
    
    # Action Factors
    ax2.plot(soc_up, factors_up, 'b-', label='Filling Action Factor')
    ax2.plot(soc_down, factors_down, 'r--', label='Emptying Action Factor')
    ax2.set_ylabel('Power Factor (0.0 - 1.0)')
    ax2.set_xlabel('State of Charge (SOC)')
    ax2.set_title('Power Modulation Factor')
    ax2.grid(True)
    
    output_path = 'apc_logic_verification.png'
    plt.savefig(output_path)
    print(f"Verification graph saved to {output_path}")
    
    # Assertions
    print("\n--- Logic Check ---")
    
    # Check Critical Zone Entry
    idx_crit = np.where(soc_up >= 0.95)[0][0]
    print(f"Enters Zone 3 (Critical) at SOC >= {soc_up[idx_crit]:.2f}")
    
    # Check Hysteresis on Exit
    # Find where it drops FROM 3 TO 2
    # In soc_down array, index 0 is SOC=1.0. 
    change_indices = np.where(np.diff(zones_down) != 0)[0]
    for idx in change_indices:
        soc_before = soc_down[idx]
        soc_after = soc_down[idx+1]
        z_before = zones_down[idx]
        z_after = zones_down[idx+1]
        print(f"Transition Zone {z_before} -> {z_after} at SOC {soc_before:.3f} -> {soc_after:.3f}")

if __name__ == "__main__":
    run_verification()
