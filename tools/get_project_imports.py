import ast
import os
import sys
from pathlib import Path

stdlib_modules = sys.builtin_module_names

def get_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
        except Exception:
            return set()
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def is_stdlib(module_name):
    # This is a basic check; stricter checks might require more extensive lists
    # but sys.stdlib_module_names is available in Python 3.10+
    if sys.version_info >= (3, 10):
        return module_name in sys.stdlib_module_names
    else:
        # Fallback for older python, though imperfect
        import imp
        try:
            imp.find_module(module_name)
            return False # heuristic: if we can find it, it might be installed, but we want to know if it's stdlib
            # simpler approach: rely on a known list or just filter out common ones manually
        except ImportError:
            return False
    return False

def main():
    root_dir = Path('/home/stuart/Documentos/Planta Hidrogenio/h2_plant')
    all_imports = set()
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                all_imports.update(get_imports(file_path))
                
    # Also scan tests
    tests_dir = Path('/home/stuart/Documentos/Planta Hidrogenio/tests')
    if tests_dir.exists():
         for root, _, files in os.walk(tests_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    all_imports.update(get_imports(file_path))

    # Filter out local packages (h2_plant)
    all_imports.discard('h2_plant')
    
    # Filter relative imports (None) - handled by split[0] usually
    
    print("Found imports:")
    sorted_imports = sorted(list(all_imports))
    for imp in sorted_imports:
        print(imp)

if __name__ == "__main__":
    main()
