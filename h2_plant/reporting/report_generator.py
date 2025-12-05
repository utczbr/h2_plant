import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Dict, Any, List, Optional

class ReportGenerator:
    """
    Generates visualization reports for the Hydrogen Plant Simulation.
    Adapted from legacy 'report.py' to support the modular architecture.
    """
    
    COLORS = {
        'offer': 'black',
        'soec': '#4CAF50',    # Green
        'pem': '#2196F3',     # Blue
        'sold': '#FF9800',    # Orange
        'price': 'black',
        'ppa': 'red',
        'limit': 'blue',
        'h2_total': 'black',
        'water_total': 'navy',
        'oxygen': 'purple',
        'tank': '#795548',    # Brown
        'compressor': '#607D8B' # Blue Grey
    }

    def __init__(self, output_dir: str = "reports", config: Dict[str, Any] = None):
        self.output_dir = output_dir
        self.config = config or {}
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('default')

    def generate_all(self, history: Dict[str, List[float]]):
        """Generate all enabled reports based on config."""
        print(f"\n--- [REPORT] Generating Reports in '{self.output_dir}' ---")
        
        # Get graph config
        graphs_config = self.config.get('visualization', {}).get('graphs', {})
        
        # --- Production ---
        if graphs_config.get('pem_h2_production_over_time', True):
            self.plot_h2_production_component(history, 'PEM')
        if graphs_config.get('soec_h2_production_over_time', True):
            self.plot_h2_production_component(history, 'SOEC')
        if graphs_config.get('total_h2_production_stacked', True):
            self.plot_h2_production_stacked(history)
        if graphs_config.get('cumulative_h2_production', True):
            self.plot_cumulative_h2(history)
        if graphs_config.get('oxygen_production_stacked', True): # NEW
            self.plot_oxygen_production(history)
        if graphs_config.get('water_consumption_stacked', True): # NEW
            self.plot_water_consumption(history)
            
        # --- Performance ---
        if graphs_config.get('pem_cell_voltage_over_time', True):
            self.plot_pem_voltage(history)
        if graphs_config.get('pem_efficiency_over_time', True):
            self.plot_pem_efficiency(history)
        if graphs_config.get('dispatch_curve_scatter', True): # NEW
            self.plot_dispatch_curve(history)
            
        # --- Economics ---
        if graphs_config.get('energy_price_over_time', True):
            self.plot_energy_price(history)
        if graphs_config.get('dispatch_strategy_stacked', True):
            self.plot_dispatch_stacked(history)
        if graphs_config.get('dispatch_detailed_overlay', True): # NEW
            self.plot_dispatch_detailed(history)
        if graphs_config.get('power_consumption_breakdown_pie', True):
            self.plot_energy_pie(history)
        if graphs_config.get('price_histogram', True): # NEW
            self.plot_price_histogram(history)
        if graphs_config.get('arbitrage_scatter', True): # NEW (Mapped from legacy plot_arbitrage)
            self.plot_arbitrage(history)
            
        # --- SOEC Ops ---
        if graphs_config.get('soec_active_modules_over_time', True):
            self.plot_soec_active_modules(history)
        if graphs_config.get('soec_modules_temporal', True): # NEW
            self.plot_modules_temporal(history)
        if graphs_config.get('soec_modules_stats', True): # NEW
            self.plot_modules_bars(history)
            
        # --- Storage ---
        if graphs_config.get('tank_storage_timeline', False):
            self.plot_tank_status(history)
        if graphs_config.get('compressor_power_stacked', True): # NEW
            self.plot_compressor_power(history)
            
        # --- Temporal Averages ---
        if graphs_config.get('temporal_averages', True):
             self.plot_temporal_averages(history)

        print("--- [REPORT] Completed ---\n")

    def _save(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close()
        print(f"   -> Saved: {filename}")

    # --- Production Graphs ---
    
    def plot_h2_production_component(self, history, component: str):
        minutes = history['minute']
        key = f'H2_{component.lower()}_kg'
        data = np.array(history.get(key, [0]*len(minutes)))
        
        plt.figure(figsize=(10, 6))
        color = self.COLORS.get(component.lower(), 'blue')
        plt.plot(minutes, data, color=color, label=f'{component} H2 Production')
        plt.title(f'{component} Hydrogen Production Rate', fontsize=14)
        plt.ylabel('Production Rate (kg/min)')
        plt.xlabel('Time (Minutes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(f'{component.lower()}_h2_production_over_time.png')

    def plot_h2_production_stacked(self, history):
        """Equivalent to legacy plot_h2_production"""
        minutes = history['minute']
        H2_soec = np.array(history.get('H2_soec_kg', [0]*len(minutes)))
        H2_pem = np.array(history.get('H2_pem_kg', [0]*len(minutes)))
        H2_total = H2_soec + H2_pem

        plt.figure(figsize=(12, 6))
        plt.fill_between(minutes, 0, H2_soec, label='H2 SOEC', color=self.COLORS['soec'], alpha=0.5)
        plt.fill_between(minutes, H2_soec, H2_total, label='H2 PEM', color=self.COLORS['pem'], alpha=0.5)
        plt.plot(minutes, H2_total, color=self.COLORS['h2_total'], linestyle='--', label='Total H2')

        plt.title('Total Hydrogen Production (Stacked)', fontsize=14)
        plt.ylabel('Production Rate (kg/min)')
        plt.xlabel('Time (Minutes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('total_h2_production_stacked.png')

    def plot_cumulative_h2(self, history):
        minutes = history['minute']
        cum_h2 = np.array(history.get('cumulative_h2_kg', [0]*len(minutes)))
        
        plt.figure(figsize=(10, 6))
        plt.plot(minutes, cum_h2, color='black', linewidth=2)
        plt.fill_between(minutes, 0, cum_h2, color='gray', alpha=0.1)
        plt.title('Cumulative Hydrogen Production', fontsize=14)
        plt.ylabel('Total H2 (kg)')
        plt.xlabel('Time (Minutes)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('cumulative_h2_production.png')

    def plot_oxygen_production(self, history):
        minutes = history['minute']
        O2_soec = np.array(history.get('O2_soec_kg', [0]*len(minutes))) # Need to log this or infer
        # If not logged, infer: H2 * 8
        if np.sum(O2_soec) == 0:
             O2_soec = np.array(history.get('H2_soec_kg', [0]*len(minutes))) * 8.0
             
        O2_pem = np.array(history.get('O2_pem_kg', [0]*len(minutes)))
        O2_total = O2_soec + O2_pem

        plt.figure(figsize=(12, 6))
        plt.fill_between(minutes, 0, O2_soec, label='O2 SOEC', color=self.COLORS['soec'], alpha=0.5)
        plt.fill_between(minutes, O2_soec, O2_total, label='O2 PEM', color=self.COLORS['oxygen'], alpha=0.5)
        plt.plot(minutes, O2_total, color='black', linestyle='--', label='Total O2')

        plt.title('Joint Oxygen Production (kg/min)', fontsize=14)
        plt.ylabel('Production Rate (kg/min)')
        plt.xlabel('Time (Minutes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('oxygen_production_stacked.png')

    def plot_water_consumption(self, history):
        minutes = history['minute']
        water_soec = np.array(history.get('steam_soec_kg', [0]*len(minutes)))
        water_pem = np.array(history.get('H2O_pem_kg', [0]*len(minutes)))
        total = water_soec + water_pem

        plt.figure(figsize=(12, 6))
        plt.fill_between(minutes, 0, water_soec, label='H2O SOEC', color=self.COLORS['soec'], alpha=0.5)
        plt.fill_between(minutes, water_soec, total, label='H2O PEM', color=self.COLORS['water_total'], alpha=0.5)
        plt.plot(minutes, total, color='black', linestyle='--', label='Total H2O')

        plt.title('Total Water Consumption', fontsize=14)
        plt.ylabel('Water Flow (kg/min)')
        plt.xlabel('Time (Minutes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('water_consumption_stacked.png')

    # --- Performance Graphs ---

    def plot_pem_voltage(self, history):
        minutes = history['minute']
        v_cell = np.array(history.get('pem_V_cell', [0]*len(minutes)))
        
        # Filter out zeros (off state) for better visualization?
        # Or just plot as is.
        
        plt.figure(figsize=(10, 6))
        plt.plot(minutes, v_cell, color=self.COLORS['pem'])
        plt.title('PEM Cell Voltage Over Time', fontsize=14)
        plt.ylabel('Voltage (V)')
        plt.xlabel('Time (Minutes)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('pem_cell_voltage_over_time.png')

    def plot_pem_efficiency(self, history):
        # Efficiency = (H2_kg * LHV) / Energy_Input
        # This is hard to calc exactly without LHV constant here.
        # Placeholder or simple calc.
        pass

    def plot_dispatch_curve(self, history):
        """Generates Dispatch Curve: H2 Produced vs Total Power."""
        P_total = np.array(history['P_soec_actual']) + np.array(history['P_pem'])
        H2_total = np.array(history.get('H2_soec_kg', 0)) + np.array(history.get('H2_pem_kg', 0))
        
        plt.figure(figsize=(10, 6))
        plt.scatter(P_total, H2_total, alpha=0.5, c='blue', edgecolors='none')
        
        plt.title('Real Dispatch Curve: H2 Production vs Power', fontsize=14)
        plt.xlabel('Total Input Power (MW)')
        plt.ylabel('H2 Production (kg/min)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('dispatch_curve_scatter.png')

    # --- Economics Graphs ---

    def plot_energy_price(self, history):
        minutes = history['minute']
        price = np.array(history.get('spot_price', [0]*len(minutes)))
        
        plt.figure(figsize=(12, 6))
        plt.plot(minutes, price, color='black')
        plt.title('Energy Price Over Time', fontsize=14)
        plt.ylabel('Price (EUR/MWh)')
        plt.xlabel('Time (Minutes)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('energy_price_over_time.png')
        
    def plot_price_histogram(self, history):
        spot_price = np.array(history['spot_price'])
        
        plt.figure(figsize=(10, 6))
        plt.hist(spot_price, bins=30, color='gray', edgecolor='black', alpha=0.7)
        
        mean_price = np.mean(spot_price)
        max_price = np.max(spot_price)
        
        plt.axvline(mean_price, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_price:.2f} EUR')
        plt.axvline(max_price, color='red', linestyle='--', linewidth=2, label=f'Max: {max_price:.2f} EUR')
        
        plt.title('Spot Price Frequency Distribution', fontsize=14)
        plt.xlabel('Price (EUR/MWh)')
        plt.ylabel('Frequency (Minutes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('price_histogram.png')

    def plot_dispatch_stacked(self, history):
        """Equivalent to legacy plot_dispatch"""
        minutes = history['minute']
        P_offer = np.array(history['P_offer'])
        P_soec = np.array(history['P_soec_actual'])
        P_pem = np.array(history['P_pem'])
        P_sold = np.array(history['P_sold'])

        plt.figure(figsize=(12, 6))
        
        plt.fill_between(minutes, 0, P_soec, label='SOEC Consumption', color=self.COLORS['soec'], alpha=0.6)
        plt.fill_between(minutes, P_soec, P_soec + P_pem, label='PEM Consumption', color=self.COLORS['pem'], alpha=0.6)
        plt.fill_between(minutes, P_soec + P_pem, P_soec + P_pem + P_sold, label='Grid Sale', color=self.COLORS['sold'], alpha=0.6)
        
        plt.plot(minutes, P_offer, label='Offered Power', color=self.COLORS['offer'], linestyle='--', linewidth=1.5)
        
        plt.title('Dispatch Strategy: Power Allocation', fontsize=14)
        plt.ylabel('Power (MW)')
        plt.xlabel('Time (Minutes)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('dispatch_strategy_stacked.png')

    def plot_dispatch_detailed(self, history):
        """Stacked Dispatch + Wind Power + Energy Price (Dual Axis)."""
        minutes = history['minute']
        P_offer = np.array(history['P_offer'])
        P_soec = np.array(history['P_soec_actual'])
        P_pem = np.array(history['P_pem'])
        P_sold = np.array(history['P_sold'])
        price = np.array(history.get('spot_price', [0]*len(minutes)))

        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Primary Axis: Power (MW)
        ax1.set_xlabel('Time (Minutes)')
        ax1.set_ylabel('Power (MW)', color='black')
        
        # Stacked Areas
        ax1.fill_between(minutes, 0, P_soec, label='SOEC Consumption', color=self.COLORS['soec'], alpha=0.6)
        ax1.fill_between(minutes, P_soec, P_soec + P_pem, label='PEM Consumption', color=self.COLORS['pem'], alpha=0.6)
        ax1.fill_between(minutes, P_soec + P_pem, P_soec + P_pem + P_sold, label='Grid Sale', color=self.COLORS['sold'], alpha=0.6)
        
        # Wind Power Line
        ax1.plot(minutes, P_offer, label='Wind Power (Turbines)', color='navy', linestyle='-', linewidth=2)
        
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        
        # Secondary Axis: Price (EUR/MWh)
        ax2 = ax1.twinx()
        color_price = 'darkblue'
        ax2.set_ylabel('Spot Price (EUR/MWh)', color=color_price)
        ax2.plot(minutes, price, color=color_price, linestyle='--', linewidth=1.5, label='Spot Price')
        ax2.tick_params(axis='y', labelcolor=color_price)
        
        # Combined Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        
        plt.title('Detailed Dispatch: Power, Wind & Price', fontsize=14, y=1.15)
        plt.tight_layout()
        self._save('dispatch_detailed_overlay.png')

    def plot_arbitrage(self, history, h2_price_eur_kg=9.6):
        minutes = history['minute']
        spot_price = np.array(history['spot_price'])
        sell_decision = np.array(history['sell_decision'])
        
        PPA_PRICE = 50.0 
        EFF_ESTIMATE_MWH_KG = 0.05 
        H2_EQUIV_PRICE = h2_price_eur_kg / EFF_ESTIMATE_MWH_KG

        plt.figure(figsize=(12, 6))
        plt.plot(minutes, spot_price, label='Spot Price (EUR/MWh)', color=self.COLORS['price'])
        plt.axhline(y=PPA_PRICE, color=self.COLORS['ppa'], linestyle='--', label='PPA Price (Contract)')
        plt.axhline(y=H2_EQUIV_PRICE, color='green', linestyle='-.', label=f'H2 Breakeven (~{H2_EQUIV_PRICE:.0f} EUR/MWh)')
        
        sell_idx = np.where(sell_decision == 1)[0]
        if len(sell_idx) > 0:
            if len(sell_idx) > 1000:
                sell_idx = sell_idx[::10]
            plt.scatter(np.array(minutes)[sell_idx], np.array(spot_price)[sell_idx], 
                       color='red', zorder=5, label='Decision: Sell', s=10)

        plt.title('Price Scenario, PPA and H2 Opportunity Cost', fontsize=14)
        plt.xlabel('Time (Minutes)')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('arbitrage_scatter.png')

    # --- SOEC Ops ---

    def plot_soec_active_modules(self, history):
        minutes = history['minute']
        active = np.array(history.get('soec_active_modules', [0]*len(minutes)))
        
        plt.figure(figsize=(10, 6))
        plt.step(minutes, active, where='post', color=self.COLORS['soec'])
        plt.title('SOEC Active Modules Over Time', fontsize=14)
        plt.ylabel('Count')
        plt.xlabel('Time (Minutes)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('soec_active_modules_over_time.png')
        
    def plot_modules_temporal(self, history):
        """Generates temporal plots for each module."""
        module_history = history.get('soec_module_powers')
        minutes = history['minute']
        
        if not module_history or len(module_history) == 0:
            print("   [!] Module data not available for temporal plots.")
            return

        data = np.array(module_history) # Shape: (Time, Modules)
        time_axis = np.array(minutes)
        num_modules = data.shape[1]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        accumulated_wear = np.cumsum(data, axis=0)
        for j in range(num_modules):
            ax1.plot(time_axis, accumulated_wear[:, j], label=f'Module {j+1}')
        
        mean_wear = np.mean(accumulated_wear, axis=1)
        ax1.plot(time_axis, mean_wear, 'k--', label='Mean', linewidth=2)
        
        ax1.set_title('Accumulated Wear per Module (MW·min)', fontsize=13)
        ax1.set_ylabel('Accumulated Energy (MW·min)')
        ax1.legend(loc='upper left', fontsize='small', ncol=2)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        for j in range(num_modules):
            ax2.plot(time_axis, data[:, j], label=f'Module {j+1}', alpha=0.8)
            
        ax2.set_title('Instantaneous Power per Module (MW)', fontsize=13)
        ax2.set_ylabel('Power (MW)')
        ax2.set_xlabel('Time (Minutes)')
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        self._save('soec_modules_temporal.png')
        
    def plot_modules_bars(self, history):
        """Generates bar charts for Total Wear."""
        module_history = history.get('soec_module_powers')
        if not module_history:
            return
            
        data = np.array(module_history)
        total_wear = np.sum(data, axis=0)
        num_modules = len(total_wear)
        x_base = np.arange(num_modules)
        labels = [f'Mod {i+1}' for i in range(num_modules)]
        
        wear_mean = np.mean(total_wear)
        
        plt.figure(figsize=(10, 6))
        plt.bar(x_base, total_wear, color='#FF9800', alpha=0.8, edgecolor='black')
        plt.axhline(wear_mean, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {wear_mean:.1f}')
        plt.title('Total Accumulated Wear (Processed Energy)', fontsize=12)
        plt.ylabel('MW·min')
        plt.xticks(x_base, labels)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save('soec_modules_stats.png')

    # --- Storage ---
    
    def plot_tank_status(self, history):
        """Generates Tank Level and Pressure plot for ALL tanks found."""
        minutes = history['minute']
        
        # Find all tank keys
        tank_keys = [k.replace('_level_kg', '') for k in history.keys() if k.endswith('_level_kg') and k != 'tank_level_kg']
        
        if not tank_keys:
            # Fallback to legacy
            tank_keys = ['tank'] 
            
        num_tanks = len(tank_keys)
        fig, axes = plt.subplots(num_tanks, 1, figsize=(12, 6 * num_tanks), sharex=True)
        if num_tanks == 1:
            axes = [axes]
            
        for i, tank_id in enumerate(tank_keys):
            ax1 = axes[i]
            
            # Try specific ID first, then legacy fallback
            level_key = f'{tank_id}_level_kg'
            pressure_key = f'{tank_id}_pressure_bar'
            
            # Handle legacy mapping if needed
            if tank_id == 'tank':
                level_key = 'tank_level_kg'
                pressure_key = 'tank_pressure_bar'
            
            level = np.array(history.get(level_key, [0]*len(minutes)))
            pressure = np.array(history.get(pressure_key, [0]*len(minutes)))
            
            color = self.COLORS['tank']
            ax1.set_ylabel(f'{tank_id}\nLevel (kg)', color=color)
            ax1.plot(minutes, level, color=color, label='Level (kg)')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Pressure (bar)', color=color)
            ax2.plot(minutes, pressure, color=color, linestyle='--', label='Pressure (bar)')
            ax2.tick_params(axis='y', labelcolor=color)
            
            ax1.set_title(f'Storage Status: {tank_id}', fontsize=12)

        axes[-1].set_xlabel('Time (Minutes)')
        plt.tight_layout()
        self._save('tank_storage_timeline.png')

    def plot_compressor_power(self, history):
        """Generates Stacked Compressor Power for ALL compressors found."""
        minutes = history['minute']
        
        # Find all compressor keys
        comp_keys = [k for k in history.keys() if k.endswith('_power_kw') and k != 'compressor_power_kw']
        
        if not comp_keys:
            return

        plt.figure(figsize=(12, 6))
        
        bottom = np.zeros(len(minutes))
        for key in comp_keys:
            data = np.array(history[key])
            label = key.replace('_power_kw', '')
            plt.fill_between(minutes, bottom, bottom + data, label=label, alpha=0.7)
            bottom += data
            
        plt.plot(minutes, bottom, 'k--', label='Total Compression Power')
        
        plt.title('Compression Power Stack (kW)', fontsize=14)
        plt.ylabel('Power (kW)')
        plt.xlabel('Time (Minutes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('compressor_power_stacked.png')

    def plot_energy_pie(self, history):
        """Generates donut chart of energy distribution."""
        E_soec = np.sum(history['P_soec_actual']) / 60.0
        E_pem = np.sum(history['P_pem']) / 60.0
        E_sold = np.sum(history['P_sold']) / 60.0
        E_total = E_soec + E_pem + E_sold
        
        sizes = [E_soec, E_pem, E_sold]
        labels = ['SOEC', 'PEM', 'Grid Sale']
        colors = [self.COLORS['soec'], self.COLORS['pem'], self.COLORS['sold']]
        
        valid_sizes = []
        valid_labels = []
        valid_colors = []
        for s, l, c in zip(sizes, labels, colors):
            if s > 0.01:
                valid_sizes.append(s)
                valid_labels.append(l)
                valid_colors.append(c)

        if not valid_sizes:
            print("   [!] Warning: No energy consumed to generate pie chart.")
            return

        fig, ax = plt.subplots(figsize=(9, 8))
        explode = [0.05 if 'Sale' in l else 0 for l in valid_labels]
        
        wedges, texts, autotexts = ax.pie(
            valid_sizes, 
            explode=explode, 
            labels=valid_labels, 
            colors=valid_colors, 
            autopct='%1.1f%%', 
            pctdistance=0.85,
            startangle=140,
            wedgeprops=dict(width=0.4, edgecolor='w'),
            textprops=dict(color="black", fontweight='bold')
        )
        
        ax.text(0, 0, f"TOTAL\n{E_total:.1f}\nMWh", ha='center', va='center', fontsize=12, fontweight='bold')
        
        plt.title('Consumed/Sold Energy Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save('report_energy_pie.png')

    def plot_temporal_averages(self, history):
        """Generates hourly/daily/monthly average plots."""
        df = pd.DataFrame(history)
        
        # Create dummy date index if not present
        start_date = "2024-01-01 00:00"
        df.index = pd.date_range(start=start_date, periods=len(df), freq='min')
        
        df['H2_total_kg'] = df.get('H2_soec_kg', 0) + df.get('H2_pem_kg', 0)
        
        periods = [
            ('hourly', 'h', 'Hourly Average'),
            ('daily', 'D', 'Daily Average'),
            ('monthly', 'ME', 'Monthly Average')
        ]
        
        for fname, freq_code, title_text in periods:
            try:
                # Select only numeric columns for resampling to avoid errors with list columns
                df_numeric = df.select_dtypes(include=[np.number])
                df_res = df_numeric.resample(freq_code).mean()
            except ValueError:
                if freq_code == 'ME':
                    df_numeric = df.select_dtypes(include=[np.number])
                    df_res = df_numeric.resample('M').mean()
                else:
                    continue

            if len(df_res) < 2:
                continue

            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # 1. Price
            axes[0].plot(df_res.index, df_res['spot_price'], color='black', marker='.', linestyle='-', linewidth=1)
            axes[0].set_ylabel('Price (EUR/MWh)')
            axes[0].set_title(f'{title_text}: Spot Market Prices')
            axes[0].grid(True, alpha=0.3)
            
            # 2. Power
            axes[1].stackplot(df_res.index, 
                              df_res['P_soec_actual'], 
                              df_res['P_pem'], 
                              df_res['P_sold'],
                              labels=['SOEC', 'PEM', 'Sold'],
                              colors=[self.COLORS['soec'], self.COLORS['pem'], self.COLORS['sold']], 
                              alpha=0.7)
            axes[1].set_ylabel('Avg Power (MW)')
            axes[1].set_title(f'{title_text}: Power Dispatch')
            axes[1].legend(loc='upper left')
            axes[1].grid(True, alpha=0.3)
            
            # 3. Production
            axes[2].plot(df_res.index, df_res['H2_total_kg'], color=self.COLORS['h2_total'], linewidth=2)
            axes[2].set_ylabel('Prod Rate (kg/min)')
            axes[2].set_title(f'{title_text}: H2 Production Rate')
            axes[2].grid(True, alpha=0.3)
            
            plt.xlabel('Simulation Time')
            plt.tight_layout()
            self._save(f'report_avg_{fname}.png')
