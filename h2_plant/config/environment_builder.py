    def _build_environment_manager(self) -> None:
        """Build and register Environment Manager for time-series data."""
        from h2_plant.components.environment.environment_manager import EnvironmentManager
        
        # Check if config specifies custom paths
        wind_path = None
        price_path = None
        
        # Environment manager always uses default paths unless specified
        env_manager = EnvironmentManager(
            wind_data_path=wind_path,
            price_data_path=price_path,
            use_default_data=True
        )
        
        self.registry.register('environment_manager', env_manager, component_type='environment')
        logger.debug("Registered Environment Manager")
