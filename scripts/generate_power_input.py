import numpy as np
import pandas as pd

# HOUR_OFFER from manager.py
HOUR_OFFER = [3.0, 5.0, 13.0, 18.0, 15.0, 9.0, 18.0, 0.0] # MW

# Expand to minute resolution (60 minutes per hour)
power_profile = []
for power in HOUR_OFFER:
    power_profile.extend([power] * 60)

# Create DataFrame
# EnvironmentManager expects columns: 'Time', 'Wind_Speed_m_s', 'Power_MW'
# Or just 'Power_MW' if we bypass wind calculation?
# Let's check EnvironmentManager._load_data logic.
# It usually reads wind speed and calculates power, OR reads power directly if available.
# But for now, let's assume it reads wind speed and we need to reverse engineer or just provide power if supported.

# Actually, EnvironmentManager usually calculates power from wind speed:
# P = 0.5 * rho * A * Cp * v^3
# To match exactly, we should modify EnvironmentManager to accept direct Power_MW column if present.

# Let's create a file with 'Power_MW' column and hope EnvironmentManager uses it or we update it.
df = pd.DataFrame({
    'Power_MW': power_profile
})

# Save as headerless CSV if that's what it expects, or with headers?
# prices_2024.csv was headerless.
# wind_data.csv usually has headers?
# Let's check wind_data.csv format or EnvironmentManager code.

# Checking EnvironmentManager code (viewed earlier):
# It likely uses pandas read_csv.

# Let's save as simple CSV with header 'Power_MW'
df.to_csv('power_input_8hour_test.csv', index=False)
print(f"Created power_input_8hour_test.csv with {len(df)} rows")
