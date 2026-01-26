
import sys
import os
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.core.cooling_manager import CoolingManager

print("PYTHON EXECUTABLE:", sys.executable)
print("CWD:", os.getcwd())
print("SYS.PATH:", sys.path)
print("DryCooler FILE:", DryCooler.__module__, DryCooler)
import inspect
print("DryCooler SOURCE FILE:", inspect.getfile(DryCooler))
print("CoolingManager SOURCE FILE:", inspect.getfile(CoolingManager))

print("\n--- INSTANTIATING DRYCOOLER ---")
dc = DryCooler(component_id="TEST_COOLER", use_central_utility=True)
print("--- INSTANTIATION COMPLETE ---\n")
