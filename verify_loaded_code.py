
import inspect
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.core.stream import Stream

print("Inspecting DryCooler.step method source:")
src = inspect.getsource(DryCooler.step)
print(src)

import ast

def find_stream_instantiation(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'id') and node.func.id == 'Stream':
                print("Found Stream instantiation:")
                print(ast.dump(node, indent=2))
                # Check keywords
                has_extra = False
                for kw in node.keywords:
                    if kw.arg == 'extra':
                        has_extra = True
                        print(f"  -> Has 'extra' argument: {ast.dump(kw.value)}")
                if not has_extra:
                    print("  -> WARNING: Missing 'extra' argument!")

find_stream_instantiation(src)
