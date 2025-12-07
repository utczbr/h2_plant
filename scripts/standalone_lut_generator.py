
"""
Standalone LUT Generator for H2 Plant Simulation (Optimized).
Run this script in any environment (Cloud, Local) with CoolProp and NumPy installed.
It generates high-resolution Lookup Tables (LUTs) compatible with the H2 Plant simulation.

Features:
- Multi-processing (Parallel execution using all CPU cores).
- Robust Fallback (Vectorized -> Loop if errors occur).
- High Resolution Config (2000x2000 points).

Requirements:
    pip install CoolProp numpy

Usage:
    python standalone_lut_generator.py
    
Output:
    Creates/Overwrites .pkl files in ./lut_cache/
"""

import os
import sys
import pickle
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Literal, Any
import time
import concurrent.futures

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [LUT Generator] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import CoolProp.CoolProp as CP
except ImportError:
    logger.error("CoolProp not found! Please run: pip install CoolProp")
    sys.exit(1)

# ==========================================
# Worker Function (Module Level for Pickle)
# ==========================================

def worker_compute_chunk(fluid: str, output: str, name1: str, val1_chunk: np.ndarray, name2: str, val2_chunk: np.ndarray):
    """
    Compute a chunk of properties. Tries vectorized call first, falls back to loop.
    Returns np.ndarray of results.
    """
    # 1. Try Vectorized (Fastest)
    try:
        # CP.PropsSI handles numpy arrays if inputs are same length
        return CP.PropsSI(output, name1, val1_chunk, name2, val2_chunk, fluid)
    except Exception:
        # 2. Fallback to Loop (Robust)
        # Often fails if one point is invalid (e.g. below min entropy).
        # We process point-by-point to salvage valid points.
        result = np.zeros_like(val1_chunk)
        for i in range(len(val1_chunk)):
            try:
                result[i] = CP.PropsSI(output, name1, val1_chunk[i], name2, val2_chunk[i], fluid)
            except:
                result[i] = 0.0 # Standard fallback for invalid region
        return result

# ==========================================
# Configuration
# ==========================================

PropertyType = Literal['D', 'H', 'S', 'C']

@dataclass
class LUTConfig:
    """Configuration for lookup table generation."""
    
    # Pressure range (Pa)
    pressure_min: float = 1e5          # 1 bar
    pressure_max: float = 1000e5       # 1000 bar
    pressure_points: int = 2000        # MAXIMUM Resolution
    
    # Temperature range (K)
    temperature_min: float = 273.15    # 0°C
    temperature_max: float = 1200.0    # ~927°C
    temperature_points: int = 2000     # MAXIMUM Resolution
    
    # Entropy range for Isentropic Lookups (J/kg/K)
    # 0.0 covers heavy fluids/liquids/two-phase starts. 100k covers hot gas H2.
    entropy_min: float = 0.0           
    entropy_max: float = 100000.0
    entropy_points: int = 500          # High resolution for isentropic
    
    # Properties to pre-compute
    properties: Tuple[PropertyType, ...] = ('D', 'H', 'S', 'C')
    
    # Fluids to support
    fluids: Tuple[str, ...] = ('H2', 'O2', 'H2O', 'N2', 'CO2', 'CH4')
    
    # Interpolation method
    interpolation: Literal['linear', 'cubic'] = 'linear'
    
    # Local Output Directory
    cache_dir: Path = Path('./lut_cache')

# ==========================================
# Generator Logic
# ==========================================

class StandaloneGenerator:
    def __init__(self, config: LUTConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        # Determine CPU count for parallelization
        self.max_workers = os.cpu_count() or 4
        
    def generate_all(self):
        logger.info("Starting Batch Generation (Parallelized)...")
        logger.info(f"Workers: {self.max_workers}")
        logger.info(f"Grid: {self.config.pressure_points}x{self.config.temperature_points} points")
        logger.info(f"Fluids: {self.config.fluids}")
        
        for fluid in self.config.fluids:
            self._generate_fluid_lut(fluid)
            
        logger.info("All LUTs generated successfully.")
        logger.info(f"Location: {self.config.cache_dir.absolute()}")

    def _generate_fluid_lut(self, fluid_name: str):
        """Generate LUT for a single fluid."""
        start_time = time.time()
        logger.info(f"Generating {fluid_name}...")
        
        # 1. Map Fluid Name to CoolProp
        cp_fluid = fluid_name
        if fluid_name == 'H2': cp_fluid = 'Hydrogen'
        if fluid_name == 'O2': cp_fluid = 'Oxygen'
        if fluid_name == 'N2': cp_fluid = 'Nitrogen'
        if fluid_name == 'H2O': cp_fluid = 'Water'
        
        lut_data = {}
        
        # 2. Main Grid Generation (P-T)
        self._compute_pt_grid(fluid_name, cp_fluid, lut_data)
        
        # 3. Isentropic Grid Generation (P-S)
        self._compute_ps_grid(fluid_name, cp_fluid, lut_data)

        # 4. Save
        self._save_lut(fluid_name, lut_data)
        
        elapsed = time.time() - start_time
        logger.info(f"  Finished {fluid_name} in {elapsed:.2f}s")

    def _parallel_compute(self, fluid, prop, name1, vals1, name2, vals2):
        """Helper to chunk and execute parallel computation."""
        # Chunk logic
        total_points = len(vals1)
        # Target ~50k-100k points per chunk for good balance of overhead/work
        chunk_size = 50000 
        chunks = []
        
        for i in range(0, total_points, chunk_size):
            end = min(i + chunk_size, total_points)
            chunks.append((
                fluid, prop, name1, vals1[i:end], name2, vals2[i:end]
            ))
            
        logger.info(f"  Computing {prop}: Splitting {total_points} pts into {len(chunks)} chunks across {self.max_workers} cores...")
        
        results_list = [None] * len(chunks)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(worker_compute_chunk, *args): idx 
                for idx, args in enumerate(chunks)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res_chunk = future.result()
                    results_list[idx] = res_chunk
                except Exception as e:
                    logger.error(f"  Chunk {idx} failed: {e}")
                    # Return zeros on catastrophic failure
                    results_list[idx] = np.zeros(len(chunks[idx][3]))

        # Concatenate
        return np.concatenate(results_list)

    def _compute_pt_grid(self, fluid_name_friendly, cp_fluid_name, lut_data):
        # Grids
        p_grid = np.linspace(self.config.pressure_min, self.config.pressure_max, self.config.pressure_points)
        t_grid = np.linspace(self.config.temperature_min, self.config.temperature_max, self.config.temperature_points)
        
        P_mesh, T_mesh = np.meshgrid(p_grid, t_grid)
        P_flat = P_mesh.flatten()
        T_flat = T_mesh.flatten()
        
        for prop in self.config.properties:
            cp_prop = prop
            if prop == 'C': cp_prop = 'Cpmass' 
            
            vals_flat = self._parallel_compute(cp_fluid_name, cp_prop, 'P', P_flat, 'T', T_flat)
            lut_data[prop] = vals_flat.reshape(P_mesh.shape)

    def _compute_ps_grid(self, fluid_name_friendly, cp_fluid_name, lut_data):
        # Grids
        p_grid = np.linspace(self.config.pressure_min, self.config.pressure_max, self.config.pressure_points)
        s_grid = np.linspace(self.config.entropy_min, self.config.entropy_max, self.config.entropy_points)
        
        P_s_mesh, S_mesh = np.meshgrid(p_grid, s_grid)
        P_s_flat = P_s_mesh.flatten()
        S_flat = S_mesh.flatten()
        
        # H(P,S)
        vals_flat = self._parallel_compute(cp_fluid_name, 'H', 'P', P_s_flat, 'S', S_flat)
        lut_data['H_from_PS'] = vals_flat.reshape(P_s_mesh.shape)

    def _save_lut(self, fluid_name, lut_data):
        safe_name = "".join(c for c in fluid_name if c.isalnum() or c in ('_', '-'))
        filename = f"lut_{safe_name}_v1.pkl"
        
        config_dict = asdict(self.config)
        if 'cache_dir' in config_dict:
            del config_dict['cache_dir']
            
        cache_data = {
            'config': config_dict,
            'lut': lut_data
        }
        
        out_path = self.config.cache_dir / filename
        with open(out_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"  Saved {filename} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    print("-" * 50)
    print(" H2 PLANT - PARALLEL LUT GENERATOR")
    print("-" * 50)
    
    config = LUTConfig()
    generator = StandaloneGenerator(config)
    generator.generate_all()
    
    print("\nGeneration Complete.")
