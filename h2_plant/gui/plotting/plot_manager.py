import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Setup logger
logger = logging.getLogger(__name__)

# Add legacy directory to path so plot_reporter_base can be found
LEGACY_PEM_DIR = Path(__file__).resolve().parents[2] / "legacy" / "NEW" / "PEM"
if str(LEGACY_PEM_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_PEM_DIR))

class LegacyPlotManager:
    """
    Orchestrates the generation of legacy plots using adapted data.
    """
    
    def __init__(self, output_dir: str = "outputs/legacy_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default Constants
        self.deoxo_mode = 'JACKET'
        self.L_deoxo = 0.6 # m
        self.dc2_mode = 'OFF'
        
    def generate_all(self, df_h2: pd.DataFrame, df_o2: pd.DataFrame, registry: Any = None):
        """
        Generate all supported legacy plots.
        
        Args:
            df_h2 (pd.DataFrame): Adapted H2 line data.
            df_o2 (pd.DataFrame): Adapted O2 line data.
            registry (ComponentRegistry): To fetch Deoxo profiles.
        """
        logger.info(f"Generating legacy plots in {self.output_dir}...")
        
        # Change CWD temporarily because plot scripts might save relative to CWD
        # or we explicitly handle saving if we can modify the reporter.
        # But the scripts call 'salvar_e_exibir_plot(filename)' which uses CWD usually
        # unless we monkeypatch it.
        # Let's try to monkeypatch 'plot_reporter_base.salvar_e_exibir_plot'
        
        # NOTE: Imports must happen AFTER patching if they import the saver function.
        original_saver = None
        
        try:
            import plot_reporter_base
            original_saver = plot_reporter_base.salvar_e_exibir_plot
            
            def patched_saver(filename, show=False):
                # Force save to our output dir
                filepath = self.output_dir / filename
                import matplotlib.pyplot as plt
                plt.savefig(filepath, dpi=300)
                plt.close()
                logger.info(f"Saved {filename}")
                
            plot_reporter_base.salvar_e_exibir_plot = patched_saver
        except ImportError:
            logger.warning("Could not patch plot reporter - plots might save in CWD")

        # Import plot modules
        
        def import_and_patch(module_name):
            import importlib
            try:
                mod = importlib.import_module(f"plots_modulos.{module_name}")
                if original_saver and hasattr(mod, 'salvar_e_exibir_plot'):
                    mod.salvar_e_exibir_plot = patched_saver
                # Assume function name matches module name
                return getattr(mod, module_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to import {module_name}: {e}")
                return None

        try:
            plot_agua_removida_total = import_and_patch('plot_agua_removida_total')
            plot_concentracao_dreno = import_and_patch('plot_concentracao_dreno')
            plot_concentracao_linha_dreno = import_and_patch('plot_concentracao_linha_dreno')
            plot_deoxo_perfil = import_and_patch('plot_deoxo_perfil')
            plot_drenos_descartados = import_and_patch('plot_drenos_descartados')
            plot_drenos_individuais = import_and_patch('plot_drenos_individuais')
            # plot_drenos_mixer = import_and_patch('plot_drenos_mixer') # Removed
            plot_esquema_drenos = import_and_patch('plot_esquema_drenos')
            plot_esquema_planta_completa = import_and_patch('plot_esquema_planta_completa')
            plot_esquema_processo = import_and_patch('plot_esquema_processo')
            plot_fluxos_energia = import_and_patch('plot_fluxos_energia')
            plot_impurezas_crossover = import_and_patch('plot_impurezas_crossover')
            plot_propriedades_empilhadas = import_and_patch('plot_propriedades_empilhadas')
            plot_propriedades_linha_dreno = import_and_patch('plot_propriedades_linha_dreno')
            plot_q_breakdown = import_and_patch('plot_q_breakdown')
            plot_recirculacao_mixer = import_and_patch('plot_recirculacao_mixer')
            plot_vazao_agua_separada = import_and_patch('plot_vazao_agua_separada')
            plot_vazao_liquida_acompanhante = import_and_patch('plot_vazao_liquida_acompanhante')
            plot_vazao_massica_total_e_removida = import_and_patch('plot_vazao_massica_total_e_removida')
            
        except Exception as e:
            logger.error(f"Failed to import/patch legacy plot modules: {e}")
            if original_saver and 'plot_reporter_base' in locals():
                plot_reporter_base.salvar_e_exibir_plot = original_saver
            return

        # Standard Args
        args = (df_h2, df_o2, self.deoxo_mode, self.L_deoxo, self.dc2_mode, False)
        
        # 1. Standard Plots
        try:
            plot_agua_removida_total(*args)
            plot_fluxos_energia(*args)
            plot_impurezas_crossover(*args)
            plot_q_breakdown(*args)
            plot_vazao_agua_separada(*args)
            plot_vazao_liquida_acompanhante(*args)
            plot_vazao_massica_total_e_removida(*args)
            
            # These might require 'df_dreno' or similar which we haven't adapted yet
            # plot_concentracao_dreno...
            # The adapter in 'generate_dataframes' returned (df_h2, df_o2).
            # Does legacy pass df_dreno? Let's check signatures if they fail.
            
        except Exception as e:
            logger.error(f"Error generating standard plots: {e}")

        # 2. Stacked Properties (Called for each fluid)
        try:
            # Args: (df, gas_fluido, deoxo_mode, L_deoxo, dc2_mode, mostrar)
            plot_propriedades_empilhadas(df_h2, 'H2', self.deoxo_mode, self.L_deoxo, self.dc2_mode, False)
            plot_propriedades_empilhadas(df_o2, 'O2', self.deoxo_mode, self.L_deoxo, self.dc2_mode, False)
        except Exception as e:
            logger.error(f"Error generating stacked properties: {e}")

        # 3. Deoxo Profile (Special)
        if registry:
            try:
                deoxo = None
                if registry.has('deoxo'):
                    deoxo = registry.get('deoxo') # Try ID
                
                if not deoxo:
                    # Search by type if ID fails
                    # We need to import strictly inside try/except to avoid circular imports if any
                    try:
                        from h2_plant.components.purification.deoxo_reactor import DeoxoReactor
                        for _, c in registry.list_components():
                            if isinstance(c, DeoxoReactor):
                                deoxo = c
                                break
                    except ImportError:
                        pass
                
                if deoxo and hasattr(deoxo, 'get_last_profiles'):
                    profiles = deoxo.get_last_profiles()
                    L_span = profiles.get('L')
                    T_prof = profiles.get('T')
                    # X_prof = profiles.get('X') # Legacy script recalculates X linearly?
                    # Let's check signature: 
                    # (df_h2, L_span, T_profile_C, X_O2, T_max_calc, deoxo_mode, L_deoxo, mostrar)
                    
                    if L_span is not None and len(L_span) > 0:
                        T_prof_C = T_prof - 273.15
                        X_O2 = deoxo.last_conversion_o2
                        T_max = deoxo.last_peak_temp_k - 273.15
                        
                        plot_deoxo_perfil(
                            df_h2, 
                            L_span, 
                            T_prof_C, 
                            X_O2, 
                            T_max, 
                            self.deoxo_mode, 
                            self.L_deoxo, 
                            False
                        )
            except Exception as e:
                logger.error(f"Error generating Deoxo profile: {e}")
                
        # Restore saver
        if original_saver and 'plot_reporter_base' in locals():
            plot_reporter_base.salvar_e_exibir_plot = original_saver
