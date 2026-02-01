"""
Pytest configuration and fixtures for h2_plant testing.

This file sets up common fixtures, test configuration, and hooks for pytest.
"""

import pytest
import tempfile
from pathlib import Path


def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup handled by tempfile automatically


@pytest.fixture(autouse=True)
def set_test_environment():
    """Set up test environment variables."""
    import os
    os.environ['H2_PLANT_ENV'] = 'test'
    yield
    # Teardown
    if 'H2_PLANT_ENV' in os.environ:
        del os.environ['H2_PLANT_ENV']


@pytest.fixture(scope="session")
def benchmark_data():
    """Provide benchmark data for performance tests."""
    import numpy as np
    # Create sample datasets for testing
    return {
        'pressure_range': np.linspace(1e5, 900e5, 100),
        'temperature_range': np.linspace(250, 350, 50),
        'mass_range': np.linspace(0, 500, 1000)
    }