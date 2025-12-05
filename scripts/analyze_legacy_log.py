import re

def analyze_log():
    max_h2 = 0.0
    max_line = ""
    
    with open('legacy_debug.log', 'r') as f:
        for line in f:
            if "| H2  |" in line:
                parts = line.split('|')
                # Format: | min | hour | ... | H2_soec | H2_pem | ...
                # Col 10 is H2_pem (index 10 if split by |)
                # Let's count:
                # 0: empty
                # 1: min
                # 2: hour
                # 3: P_offer
                # 4: P_soec_set
                # 5: P_soec_actual
                # 6: P_pem
                # 7: P_sold
                # 8: spot_price
                # 9: " H2  "
                # 10: h2_soec
                # 11: h2_pem
                
                try:
                    h2_pem = float(parts[11].strip())
                    if h2_pem > max_h2:
                        max_h2 = h2_pem
                        max_line = line
                except:
                    pass
                    
    print(f"Max PEM H2: {max_h2} kg/min")
    print(f"Line: {max_line}")

if __name__ == "__main__":
    analyze_log()
