"""
New build methods for detailed system configurations (to be added to PlantBuilder).
"""

def _build_pem_system(self) -> None:
    """Build PEM electrolysis system from detailed component configuration."""
    if not self.config.pem_system:
        return
    
    pem = self.config.pem_system
    logger.info("Building PEM system from detailed components")
    
    # Import component classes
    from h2_plant.components.electrolysis.pem_stack import PEMStack
    from h2_plant.components.electrolysis.rectifier import RectifierTransformer
    from h2_plant.components.thermal.heat_exchanger import HeatExchanger
    from h2_plant.components.fluid.recirculation_pump import RecirculationPump
    from h2_plant.components.separation.psa_unit import PSAUnit
    from h2_plant.components.separation.separation_tank import SeparationTank
    
    # Build stacks
    for stack_cfg in pem.get('stacks', []):
        stack = PEMStack(
            max_power_kw=stack_cfg.get('max_power_kw', 2500.0),
            cells_per_stack=stack_cfg.get('cells_per_stack', 85),
            parallel_stacks=stack_cfg.get('parallel_stacks', 36),
            active_area_m2=stack_cfg.get('active_area_m2', 0.03)
        )
        component_id = stack_cfg.get('component_id', f'pem_stack_{len(self.registry.get_by_type("pem_production"))}')
        self.registry.register(component_id, stack, component_type='pem_production')
        logger.debug(f"Registered PEM stack: {component_id}")
    
    # Build rectifiers
    for rect_cfg in pem.get('rectifiers', []):
        rectifier = RectifierTransformer(
            max_power_kw=rect_cfg.get('max_power_kw', 2500.0),
            efficiency=rect_cfg.get('efficiency', 0.98)
        )
        component_id = rect_cfg.get('component_id', f'pem_rectifier_{len(self.registry.get_by_type("pem_power"))}')
        self.registry.register(component_id, rectifier, component_type='pem_power')
    
    # Build heat exchangers
    for hx_cfg in pem.get('heat_exchangers', []):
        hx = HeatExchanger(
            component_id=hx_cfg.get('component_id', 'HX-PEM'),
            max_heat_removal_kw=hx_cfg.get('max_heat_removal_kw', 500.0),
            target_outlet_temp_c=hx_cfg.get('target_outlet_temp_c', 25.0)
        )
        self.registry.register(hx.component_id, hx, component_type='pem_thermal')
    
    # Build pumps
    for pump_cfg in pem.get('pumps', []):
        from h2_plant.components.water.pump import WaterPump
        pump = WaterPump(
            pump_id=pump_cfg.get('component_id', 'P-PEM'),
            power_kw=10.0,  # Simplified
            power_source='grid',
            outlet_pressure_bar=pump_cfg.get('pressure_bar', 5.0)
        )
        self.registry.register(pump.pump_id, pump, component_type='pem_water')
    
    # Build separation tanks
    for tank_cfg in pem.get('separation_tanks', []):
        tank = SeparationTank(
            component_id=tank_cfg.get('component_id', 'ST-PEM'),
            gas_type=tank_cfg.get('gas_type', 'H2')
        )
        self.registry.register(tank.component_id, tank, component_type='pem_separation')
    
    # Build PSA units
    for psa_cfg in pem.get('psa_units', []):
        psa = PSAUnit(
            component_id=psa_cfg.get('component_id', 'PSA-PEM'),
            gas_type=psa_cfg.get('gas_type', 'H2')
        )
        self.registry.register(psa.component_id, psa, component_type='pem_separation')

def _build_soec_system(self) -> None:
    """Build SOEC electrolysis system from detailed component configuration."""
    if not self.config.soec_system:
        return
    
    soec = self.config.soec_system
    logger.info("Building SOEC system from detailed components")
    
    from h2_plant.components.electrolysis.soec_stack import SOECStack
    from h2_plant.components.electrolysis.rectifier import RectifierTransformer
    from h2_plant.components.reforming.steam_generator import SteamGenerator
    
    # Build SOEC stacks
    for stack_cfg in soec.get('stacks', []):
        stack = SOECStack(max_power_kw=stack_cfg.get('max_power_kw', 1000.0))
        component_id = stack_cfg.get('component_id', 'soec_stack_1')
        self.registry.register(component_id, stack, component_type='soec_production')
    
    # Build rectifiers
    for rect_cfg in soec.get('rectifiers', []):
        rectifier = RectifierTransformer(
            max_power_kw=rect_cfg.get('max_power_kw', 1000.0),
            efficiency=rect_cfg.get('efficiency', 0.98)
        )
        self.registry.register(rect_cfg.get('component_id', 'RT-SOEC'), rectifier, component_type='soec_power')
    
    # Build steam generators
    for sg_cfg in soec.get('steam_generators', []):
        sg = SteamGenerator(
            component_id=sg_cfg.get('component_id', 'SG-SOEC'),
            max_flow_kg_h=sg_cfg.get('max_flow_kg_h', 500.0)
        )
        self.registry.register(sg.component_id, sg, component_type='soec_thermal')

def _build_atr_system(self) -> None:
    """Build ATR reforming system from detailed component configuration."""
    if not self.config.atr_system:
        return
    
    atr = self.config.atr_system
    logger.info("Building ATR system from detailed components")
    
    from h2_plant.components.reforming.atr_reactor import ATRReactor
    from h2_plant.components.reforming.wgs_reactor import WGSReactor
    
    # Build ATR reactors
    for reactor_cfg in atr.get('reactors', []):
        reactor = ATRReactor(
            component_id=reactor_cfg.get('component_id', 'ATR-1'),
            max_flow_kg_h=reactor_cfg.get('max_flow_kg_h', 1500.0),
            model_path=reactor_cfg.get('model_path', 'to_integrate/ATR_model_functions.pkl')
        )
        self.registry.register(reactor.component_id, reactor, component_type='atr_production')
    
    # Build WGS reactors
    for wgs_cfg in atr.get('wgs_reactors', []):
        wgs = WGSReactor(
            component_id=wgs_cfg.get('component_id', 'WGS-HT'),
            conversion_rate=wgs_cfg.get('conversion_rate', 0.7)
        )
        self.registry.register(wgs.component_id, wgs, component_type='atr_conversion')

def _build_logistics(self) -> None:
    """Build logistics components (consumers/refueling stations)."""
    if not self.config.logistics:
        return
    
    logistics = self.config.logistics
    logger.info("Building logistics components")
    
    from h2_plant.components.logistics.consumer import Consumer
    
    for consumer_cfg in logistics.get('consumers', []):
        consumer = Consumer(
            num_bays=consumer_cfg.get('num_bays', 4),
            filling_rate_kg_h=consumer_cfg.get('filling_rate_kg_h', 50.0)
        )
        component_id = consumer_cfg.get('component_id', 'consumer_1')
        self.registry.register(component_id, consumer, component_type='logistics')
