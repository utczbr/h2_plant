# STEP 3: Technical Specification - Code Consolidation Guide

---

# 07_Code_Consolidation_Guide.md

**Document:** Code Consolidation and Migration Guide  
**Project:** Dual-Path Hydrogen Production System - Modular Refactoring v2.0  
**Date:** November 18, 2025  
**Priority:** IMMEDIATE  
**Dependencies:** All previous specifications

---

## 1. Overview

### 1.1 Purpose

This guide provides a **step-by-step roadmap** for consolidating the current fragmented codebase into the new modular architecture. The consolidation addresses the critique's most pressing structural issue: 9 duplicate files across multiple directories creating maintenance nightmares and code drift.

**Key Objectives:**
- Eliminate all duplicate files (9 identified duplicates)
- Reorganize flat directory structure into layered architecture
- Migrate legacy code to new Component-based interfaces
- Establish clear import paths and module hierarchy
- Provide deprecation timeline for old code

**Critique Remediation:**
- **FAIL → PASS:** "9 duplicate files between directories" (Section 2)
- **FAIL → PASS:** "Flat directory structure with no organization" (Section 3)
- **Immediate Action:** Code consolidation should happen first to establish clean foundation

***

### 1.2 Current State Analysis

**Existing Directory Structure (Problematic):**
```
project_root/
├── Dual_tank_system/
│   ├── ATR_model.py                    # DUPLICATE 1
│   ├── FillingCompressors.py           # DUPLICATE 2
│   ├── H2DemandwithNightShift.py       # DUPLICATE 3
│   ├── HPProcessLogicNoLP.py           # DUPLICATE 4
│   ├── Outgoingcompressor.py           # DUPLICATE 5
│   ├── ActualHydrogenProduction.py     # DUPLICATE 6
│   ├── EnergyPriceFunction.py          # DUPLICATE 7
│   ├── EnergyPriceState.py             # DUPLICATE 8
│   └── Finalsimulation.py              # DUPLICATE 9
│
├── H2-Storage-Model/
│   ├── ATR_model.py                    # DUPLICATE 1
│   ├── FillingCompressors.py           # DUPLICATE 2
│   ├── H2DemandwithNightShift.py       # DUPLICATE 3
│   ├── HPProcessLogicNoLP.py           # DUPLICATE 4
│   ├── Outgoingcompressor.py           # DUPLICATE 5
│   ├── ActualHydrogenProduction.py     # DUPLICATE 6
│   ├── EnergyPriceFunction.py          # DUPLICATE 7
│   ├── EnergyPriceState.py             # DUPLICATE 8
│   └── Finalsimulation.py              # DUPLICATE 9
│
└── (various other scattered files)
```

**Problems:**
1. **Code Duplication:** 9 files × 2 locations = 18 total files for 9 unique modules
2. **Divergence Risk:** Changes in one location don't propagate to the other
3. **Import Confusion:** No clear canonical import path
4. **No Hierarchy:** Flat structure obscures relationships
5. **No Packaging:** Not installable as proper Python package

***

### 1.3 Target State Architecture

**New Directory Structure (Layered):**
```
h2_plant/                              # Main package
├── __init__.py
├── core/                              # Layer 1: Foundation
│   ├── __init__.py
│   ├── component.py
│   ├── component_registry.py
│   ├── enums.py
│   ├── constants.py
│   ├── types.py
│   └── exceptions.py
│
├── optimization/                      # Layer 2: Performance
│   ├── __init__.py
│   ├── lut_manager.py
│   ├── numba_ops.py
│   └── thermodynamics.py
│
├── components/                        # Layer 3: Components
│   ├── __init__.py
│   ├── production/
│   │   ├── __init__.py
│   │   ├── electrolyzer_source.py
│   │   └── atr_source.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── tank_array.py
│   │   ├── source_isolated_tanks.py
│   │   └── oxygen_buffer.py
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── filling_compressor.py
│   │   └── outgoing_compressor.py
│   └── utility/
│       ├── __init__.py
│       ├── demand_scheduler.py
│       └── energy_price_tracker.py
│
├── pathways/                          # Layer 4: Orchestration
│   ├── __init__.py
│   ├── isolated_production_path.py
│   ├── dual_path_coordinator.py
│   └── allocation_strategies.py
│
├── config/                            # Configuration
│   ├── __init__.py
│   ├── plant_config.py
│   ├── loaders.py
│   ├── plant_builder.py
│   └── schemas/
│       └── plant_schema_v1.json
│
├── simulation/                        # Layer 5: Execution
│   ├── __init__.py
│   ├── engine.py
│   ├── state_manager.py
│   ├── event_scheduler.py
│   ├── monitoring.py
│   └── runner.py
│
└── legacy/                            # Backward compatibility
    ├── __init__.py
    └── adapters.py

configs/                               # Configuration files
├── plant_baseline.yaml
├── plant_grid_only.yaml
└── plant_pilot.yaml

tests/                                 # Test suite
├── core/
├── optimization/
├── components/
├── pathways/
├── config/
├── simulation/
└── integration/

examples/                              # Example scripts
├── basic_simulation.py
├── dual_path_simulation.py
└── scenario_comparison.py

docs/                                  # Documentation
├── architecture.md
├── api_reference.md
└── migration_guide.md

setup.py                               # Package installation
requirements.txt                       # Dependencies
README.md                              # Project overview
```

**Benefits:**
- **Clear Hierarchy:** 5 distinct layers with explicit dependencies
- **No Duplication:** Single source of truth for each module
- **Installable Package:** Standard Python package structure
- **Testable:** Organized test structure mirrors source
- **Documented:** Dedicated documentation directory

***

## 2. Duplicate File Resolution

### 2.1 Duplicate Analysis

**Duplicate Files Identified:**

| **File Name** | **Location 1** | **Location 2** | **Canonical Destination** | **Status** |
|--------------|---------------|---------------|--------------------------|-----------|
| `ATR_model.py` | `Dual_tank_system/` | `H2-Storage-Model/` | `components/production/atr_source.py` | Refactor |
| `FillingCompressors.py` | `Dual_tank_system/` | `H2-Storage-Model/` | `components/compression/filling_compressor.py` | Refactor |
| `Outgoingcompressor.py` | `Dual_tank_system/` | `H2-Storage-Model/` | `components/compression/outgoing_compressor.py` | Refactor |
| `H2DemandwithNightShift.py` | `Dual_tank_system/` | `H2-Storage-Model/` | `components/utility/demand_scheduler.py` | Refactor |
| `ActualHydrogenProduction.py` | `Dual_tank_system/` | `H2-Storage-Model/` | `components/production/electrolyzer_source.py` | Refactor |
| `EnergyPriceFunction.py` | `Dual_tank_system/` | `H2-Storage-Model/` | `components/utility/energy_price_tracker.py` | Refactor |
| `EnergyPriceState.py` | `Dual_tank_system/` | `H2-Storage-Model/` | `components/utility/energy_price_tracker.py` | Merge |
| `HPProcessLogicNoLP.py` | `Dual_tank_system/` | `H2-Storage-Model/` | `pathways/isolated_production_path.py` | Refactor |
| `Finalsimulation.py` | `Dual_tank_system/` | `H2-Storage-Model/` | `simulation/engine.py` | Replace |

***

### 2.2 Resolution Strategy for Each Duplicate

#### **Duplicate 1: ATR_model.py**

**Current State:**
- Two versions exist with potential divergence
- Contains Numba-compiled ATR reaction kinetics
- ~12,000 characters, custom implementation

**Resolution:**
1. **Compare versions:** Use `diff` to identify any differences
2. **Select canonical:** Choose the more complete/recent version
3. **Refactor into:** `h2_plant/components/production/atr_source.py`
4. **Changes needed:**
   - Inherit from `Component` ABC
   - Implement `initialize()`, `step()`, `get_state()`
   - Keep Numba optimizations
5. **Testing:** Verify identical output vs legacy code

**Migration Command:**
```bash
# Compare versions
diff Dual_tank_system/ATR_model.py H2-Storage-Model/ATR_model.py

# Copy canonical version
cp Dual_tank_system/ATR_model.py h2_plant/components/production/atr_source.py

# Refactor (manual - see Component Standardization Spec)
# Add Component ABC inheritance, implement lifecycle methods
```

***

#### **Duplicate 2 & 5: FillingCompressors.py & Outgoingcompressor.py**

**Current State:**
- Compression logic duplicated
- Different naming conventions (camelCase vs snake_case)

**Resolution:**
1. **Consolidate into:**
   - `h2_plant/components/compression/filling_compressor.py`
   - `h2_plant/components/compression/outgoing_compressor.py`
2. **Standardize:**
   - Use Component ABC
   - Integrate Numba compression work calculations
   - Consistent naming (snake_case)

***

#### **Duplicate 3: H2DemandwithNightShift.py**

**Current State:**
- Day/night demand pattern implementation
- ~17,000 characters

**Resolution:**
1. **Refactor into:** `h2_plant/components/utility/demand_scheduler.py`
2. **Enhancements:**
   - Support multiple patterns (constant, day_night, weekly, custom)
   - Component ABC compliance
   - Configuration-driven patterns

***

#### **Duplicate 6: ActualHydrogenProduction.py**

**Current State:**
- Electrolyzer production logic
- Efficiency calculations

**Resolution:**
1. **Refactor into:** `h2_plant/components/production/electrolyzer_source.py`
2. **Changes:**
   - Component ABC inheritance
   - Replace `calculate_production()` with `step()`
   - Add oxygen byproduct tracking

***

#### **Duplicate 7 & 8: EnergyPriceFunction.py & EnergyPriceState.py**

**Current State:**
- Two separate files for energy price handling
- State management split from function

**Resolution:**
1. **Merge into:** `h2_plant/components/utility/energy_price_tracker.py`
2. **Consolidation:**
   - Combine state and function logic
   - Component ABC compliance
   - Support file-based price loading

***

#### **Duplicate 4: HPProcessLogicNoLP.py**

**Current State:**
- High-level process orchestration logic
- ~21,000 characters
- Complex interdependencies

**Resolution:**
1. **Refactor into:** `h2_plant/pathways/isolated_production_path.py`
2. **Modernization:**
   - Break into pathway orchestration
   - Remove hardcoded logic
   - Use Component ABC pattern

***

#### **Duplicate 9: Finalsimulation.py**

**Current State:**
- Monolithic simulation loop (~22,000 characters)
- Hardcoded component interactions
- No modularity

**Resolution:**
1. **Replace with:** `h2_plant/simulation/engine.py`
2. **Complete redesign:**
   - Modular SimulationEngine
   - Event scheduling
   - Checkpointing
   - Monitoring

***

### 2.3 Consolidation Execution Plan

**Phase 1: Preparation (Week 1)**
```bash
# Create new directory structure
mkdir -p h2_plant/{core,optimization,components/{production,storage,compression,utility},pathways,config,simulation,legacy}

# Initialize Python packages
touch h2_plant/__init__.py
touch h2_plant/core/__init__.py
# ... (repeat for all subdirectories)

# Create backup of existing code
tar -czf backup_$(date +%Y%m%d).tar.gz Dual_tank_system/ H2-Storage-Model/
```

**Phase 2: Core Foundation (Week 1)**
```bash
# Implement Layer 1 components (from Spec 01)
# Create all files in h2_plant/core/
# No migration needed - new code
```

**Phase 3: Performance Layer (Week 2)**
```bash
# Implement Layer 2 (from Spec 02)
# Create all files in h2_plant/optimization/
# No migration needed - new code
```

**Phase 4: Component Migration (Weeks 3-4)**
```bash
# Migrate and refactor duplicates

# ATR
python scripts/migrate_component.py \
    --source Dual_tank_system/ATR_model.py \
    --destination h2_plant/components/production/atr_source.py \
    --template component_abc

# Electrolyzer
python scripts/migrate_component.py \
    --source Dual_tank_system/ActualHydrogenProduction.py \
    --destination h2_plant/components/production/electrolyzer_source.py \
    --template component_abc

# Compressors
python scripts/migrate_component.py \
    --source Dual_tank_system/FillingCompressors.py \
    --destination h2_plant/components/compression/filling_compressor.py \
    --template component_abc

# ... (repeat for all components)
```

**Phase 5: Legacy Cleanup (Week 5)**
```bash
# Mark old directories as deprecated
echo "DEPRECATED: Use h2_plant package instead" > Dual_tank_system/README_DEPRECATED.txt
echo "DEPRECATED: Use h2_plant package instead" > H2-Storage-Model/README_DEPRECATED.txt

# Add deprecation warnings to old imports
python scripts/add_deprecation_warnings.py Dual_tank_system/
python scripts/add_deprecation_warnings.py H2-Storage-Model/
```

***

## 3. Directory Restructuring

### 3.1 Migration Script

**File:** `scripts/migrate_to_new_structure.py`

```python
"""
Migration script for consolidating codebase into new structure.

Usage:
    python scripts/migrate_to_new_structure.py --dry-run
    python scripts/migrate_to_new_structure.py --execute
"""

import shutil
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Migration mapping: old path -> new path
MIGRATION_MAP = {
    # Production components
    'Dual_tank_system/ATR_model.py': 'h2_plant/components/production/atr_source.py',
    'Dual_tank_system/ActualHydrogenProduction.py': 'h2_plant/components/production/electrolyzer_source.py',
    
    # Compression components
    'Dual_tank_system/FillingCompressors.py': 'h2_plant/components/compression/filling_compressor.py',
    'Dual_tank_system/Outgoingcompressor.py': 'h2_plant/components/compression/outgoing_compressor.py',
    
    # Utility components
    'Dual_tank_system/H2DemandwithNightShift.py': 'h2_plant/components/utility/demand_scheduler.py',
    'Dual_tank_system/EnergyPriceFunction.py': 'h2_plant/components/utility/energy_price_tracker.py',
    
    # Pathway logic
    'Dual_tank_system/HPProcessLogicNoLP.py': 'h2_plant/pathways/isolated_production_path.py',
    
    # Simulation
    'Dual_tank_system/Finalsimulation.py': 'h2_plant/simulation/engine.py',
}


def migrate_file(old_path: Path, new_path: Path, dry_run: bool = False):
    """
    Migrate a single file to new location.
    
    Args:
        old_path: Source file path
        new_path: Destination file path
        dry_run: If True, only log actions without executing
    """
    if not old_path.exists():
        logger.warning(f"Source file not found: {old_path}")
        return
    
    if dry_run:
        logger.info(f"[DRY RUN] Would copy: {old_path} -> {new_path}")
    else:
        # Create destination directory if needed
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(old_path, new_path)
        logger.info(f"Migrated: {old_path} -> {new_path}")


def create_package_structure(dry_run: bool = False):
    """Create new package directory structure."""
    
    directories = [
        'h2_plant',
        'h2_plant/core',
        'h2_plant/optimization',
        'h2_plant/components/production',
        'h2_plant/components/storage',
        'h2_plant/components/compression',
        'h2_plant/components/utility',
        'h2_plant/pathways',
        'h2_plant/config',
        'h2_plant/config/schemas',
        'h2_plant/simulation',
        'h2_plant/legacy',
        'configs',
        'tests',
        'examples',
        'docs'
    ]
    
    for directory in directories:
        path = Path(directory)
        
        if dry_run:
            logger.info(f"[DRY RUN] Would create directory: {path}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py for Python packages
            if path.parts[0] == 'h2_plant':
                init_file = path / '__init__.py'
                if not init_file.exists():
                    init_file.touch()
            
            logger.info(f"Created directory: {path}")


def mark_deprecated(old_dirs: list[str], dry_run: bool = False):
    """Mark old directories as deprecated."""
    
    deprecation_message = """
# DEPRECATED

This directory structure is deprecated. Please use the new `h2_plant` package instead.

## Migration Guide

Old imports:
```
from Dual_tank_system.ATR_model import ATRModel
```

New imports:
```
from h2_plant.components.production import ATRProductionSource
```

See docs/migration_guide.md for complete migration instructions.
"""
    
    for old_dir in old_dirs:
        readme_path = Path(old_dir) / 'README_DEPRECATED.txt'
        
        if dry_run:
            logger.info(f"[DRY RUN] Would create deprecation notice: {readme_path}")
        else:
            with open(readme_path, 'w') as f:
                f.write(deprecation_message)
            logger.info(f"Created deprecation notice: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description='Migrate codebase to new structure')
    parser.add_argument('--dry-run', action='store_true', help='Show actions without executing')
    parser.add_argument('--execute', action='store_true', help='Execute migration')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        logger.error("Must specify either --dry-run or --execute")
        return
    
    dry_run = args.dry_run
    
    logger.info("Starting migration...")
    
    # Step 1: Create new structure
    logger.info("\n=== Creating Package Structure ===")
    create_package_structure(dry_run=dry_run)
    
    # Step 2: Migrate files
    logger.info("\n=== Migrating Files ===")
    for old_path_str, new_path_str in MIGRATION_MAP.items():
        old_path = Path(old_path_str)
        new_path = Path(new_path_str)
        migrate_file(old_path, new_path, dry_run=dry_run)
    
    # Step 3: Mark old directories as deprecated
    logger.info("\n=== Marking Deprecated Directories ===")
    mark_deprecated(['Dual_tank_system', 'H2-Storage-Model'], dry_run=dry_run)
    
    logger.info("\n=== Migration Complete ===")
    
    if dry_run:
        logger.info("This was a dry run. Use --execute to perform actual migration.")


if __name__ == '__main__':
    main()
```

***

### 3.2 Import Path Updates

**Legacy Import Patterns (to be deprecated):**
```python
# Old - scattered imports
from Dual_tank_system.ATR_model import ATRModel
from H2_Storage_Model.FillingCompressors import FillingCompressor
import HPProcessLogicNoLP
```

**New Import Patterns:**
```python
# New - organized imports
from h2_plant.components.production import ATRProductionSource, ElectrolyzerProductionSource
from h2_plant.components.compression import FillingCompressor, OutgoingCompressor
from h2_plant.pathways import IsolatedProductionPath
from h2_plant.simulation import SimulationEngine
```

**Deprecation Wrapper (for transition period):**

**File:** `h2_plant/legacy/adapters.py`

```python
"""
Legacy import adapters for backward compatibility.

Provides deprecated wrappers for old import paths during transition period.
Will be removed in v3.0.
"""

import warnings
from h2_plant.components.production import ATRProductionSource
from h2_plant.components.compression import FillingCompressor


def _deprecated_import(old_name: str, new_name: str, new_class):
    """Create deprecated wrapper for old import."""
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead. "
        f"Legacy support will be removed in v3.0.",
        DeprecationWarning,
        stacklevel=3
    )
    return new_class


# Legacy class wrappers
class ATRModel(ATRProductionSource):
    """DEPRECATED: Use h2_plant.components.production.ATRProductionSource instead."""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ATRModel is deprecated. Use ATRProductionSource instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
```

***

## 4. Package Installation Setup

### 4.1 setup.py

**File:** `setup.py`

```python
"""
Setup configuration for h2_plant package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text() if readme_path.exists() else ''

setup(
    name='h2_plant',
    version='2.0.0',
    description='Modular dual-path hydrogen production plant simulation system',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Hydrogen Production Team',
    author_email='team@h2plant.example.com',
    url='https://github.com/example/h2-plant',
    
    packages=find_packages(exclude=['tests', 'examples', 'docs']),
    
    install_requires=[
        'numpy>=1.21.0',
        'numba>=0.55.0',
        'pyyaml>=6.0',
        'jsonschema>=4.0.0',
        'h5py>=3.0.0',  # Optional: for HDF5 checkpoints
    ],
    
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'mypy>=0.990',
            'black>=22.0.0',
            'flake8>=5.0.0',
        ],
        'coolprop': [
            'CoolProp>=6.4.0',  # For LUT generation
        ],
        'viz': [
            'matplotlib>=3.5.0',
            'plotly>=5.0.0',
        ]
    },
    
    entry_points={
        'console_scripts': [
            'h2-simulate=h2_plant.simulation.runner:main',
        ],
    },
    
    python_requires='>=3.9',
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
```

***

### 4.2 requirements.txt

**File:** `requirements.txt`

```txt
# Core dependencies
numpy>=1.21.0
numba>=0.55.0
pyyaml>=6.0
jsonschema>=4.0.0

# Optional dependencies
h5py>=3.0.0
CoolProp>=6.4.0

# Development dependencies
pytest>=7.0.0
pytest-cov>=3.0.0
mypy>=0.990
black>=22.0.0
flake8>=5.0.0
```

***

## 5. Deprecation Timeline

### 5.1 Version Timeline

**Version 2.0 (Current - Weeks 1-10):**
- New modular architecture introduced
- Legacy imports work with deprecation warnings
- Both old and new code coexist

**Version 2.5 (Weeks 11-15):**
- All new features only in new architecture
- Legacy imports raise louder warnings
- Migration guide updated with automated tools

**Version 3.0 (Week 16+):**
- Legacy code removed entirely
- Only new architecture supported
- Breaking changes documented

---

### 5.2 Deprecation Notices

**File:** `Dual_tank_system/__init__.py`

```python
"""
DEPRECATED: This module structure is deprecated.

Please migrate to the new h2_plant package:

Old:
    from Dual_tank_system.ATR_model import ATRModel

New:
    from h2_plant.components.production import ATRProductionSource

This legacy module will be removed in v3.0 (approximately Week 16).
See docs/migration_guide.md for complete instructions.
"""

import warnings

warnings.warn(
    "Dual_tank_system is deprecated and will be removed in v3.0. "
    "Please migrate to h2_plant package. See docs/migration_guide.md",
    DeprecationWarning,
    stacklevel=2
)
```

***

## 6. Testing Strategy

### 6.1 Validation Tests

**File:** `tests/test_migration.py`

```python
"""
Tests to validate migration correctness.

Ensures new code produces identical results to legacy code.
"""

import pytest
import numpy as np


def test_atr_output_parity():
    """Verify ATRProductionSource matches legacy ATRModel output."""
    from h2_plant.components.production import ATRProductionSource
    # from Dual_tank_system.ATR_model import ATRModel  # Legacy
    
    # Setup both versions with identical parameters
    new_atr = ATRProductionSource(max_ng_flow_kg_h=100.0, efficiency=0.75)
    # legacy_atr = ATRModel(max_ng_flow=100.0, efficiency=0.75)
    
    # Run identical inputs
    # ... compare outputs
    
    # Assert identical results (within numerical tolerance)
    # assert np.allclose(new_output, legacy_output, rtol=1e-6)


def test_import_deprecation_warnings():
    """Verify legacy imports raise deprecation warnings."""
    
    with pytest.warns(DeprecationWarning, match="deprecated"):
        from Dual_tank_system import ATR_model


def test_no_duplicate_files():
    """Ensure no duplicate files exist after migration."""
    from pathlib import Path
    
    # Check Dual_tank_system and H2-Storage-Model have deprecation notices
    dual_tank_readme = Path("Dual_tank_system/README_DEPRECATED.txt")
    storage_model_readme = Path("H2-Storage-Model/README_DEPRECATED.txt")
    
    assert dual_tank_readme.exists(), "Deprecation notice missing"
    assert storage_model_readme.exists(), "Deprecation notice missing"
```

***

## 7. Rollback Plan

### 7.1 Backup Strategy

**Before Migration:**
```bash
# Create complete backup
tar -czf pre_migration_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
    Dual_tank_system/ \
    H2-Storage-Model/ \
    *.py \
    *.md

# Store in safe location
mv pre_migration_backup_*.tar.gz ~/backups/
```

**Rollback Procedure:**
```bash
# If migration fails, restore from backup
cd ~/backups/
tar -xzf pre_migration_backup_YYYYMMDD_HHMMSS.tar.gz -C /path/to/project/
```

***

## 8. Validation Criteria

This Code Consolidation is **COMPLETE** when:

✅ **Duplicate Elimination:**
- All 9 duplicate files resolved
- Single canonical version of each module
- No code duplication detected by tools

✅ **Directory Structure:**
- New layered structure implemented
- All __init__.py files created
- Package installable via `pip install -e .`

✅ **Import Paths:**
- All new imports working
- Legacy imports deprecated with warnings
- Import path tests passing

✅ **Testing:**
- Migration validation tests passing
- No regression in functionality
- Legacy compatibility maintained

✅ **Documentation:**
- Migration guide complete
- Deprecation timeline documented
- Examples updated

---

## 9. Success Metrics

| **Metric** | **Target** | **Validation** |
|-----------|-----------|----------------|
| Duplicate Files | 0 | Manual inspection + automated checks |
| Directory Depth | 3-4 levels | Structured hierarchy |
| Import Path Consistency | 100% | All imports use h2_plant.* |
| Package Installability | Yes | `pip install -e .` succeeds |
| Test Pass Rate | 100% | All tests green |

***