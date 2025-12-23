import os
import re
import subprocess

# Mapping of PyPI package name to detected import name(s)
PACKAGE_TO_IMPORTS = {
    "attrs": ["attr", "attrs"],
    "beautifulsoup4": ["bs4"],
    "CoolProp": ["CoolProp"],
    "entsoe-py": ["entsoe"],
    "pydantic": ["pydantic"],
    "PySide6": ["PySide6"],
    "PyYAML": ["yaml"],
    "scikit-learn": ["sklearn"],
    "Pillow": ["PIL"],
    "scipy": ["scipy"],
    "matplotlib": ["matplotlib", "pylab", "pyplot"],
    "numpy": ["numpy"],
    "pandas": ["pandas"],
    "plotly": ["plotly"],
    "requests": ["requests"],
    "xlrd": ["xlrd"],
    "windpowerlib": ["windpowerlib"],
    "typing-extensions": ["typing_extensions"],
    "typing_extensions": ["typing_extensions"],
    "python-dateutil": ["dateutil"],
    "pytz": ["pytz"],
    "numba": ["numba"],
    "NodeGraphQt": ["NodeGraphQt"],
    "h5py": ["h5py"],
    "invoke": ["invoke"],
    "click": ["click"],
    "black": ["black"],
    "flake8": ["flake8"],
    "mypy": ["mypy"],
    "pytest": ["pytest"],
    "coverage": ["coverage"],
    "PyJWT": ["jwt"],
    "PyNaCl": ["nacl"],
    "pycairo": ["cairo"],
    "pycodestyle": ["pycodestyle"],
    "pyflakes": ["pyflakes"],
    "Pygments": ["pygments"],
    "oauthlib": ["oauthlib"],
    "packaging": ["packaging"],
    "paramiko": ["paramiko"],
    "pathspec": ["pathspec"],
    "pexpect": ["pexpect"],
    "phonenumbers": ["phonenumbers"],
    "platformdirs": ["platformdirs"],
    "pluggy": ["pluggy"],
    "psutil": ["psutil"],
    "ptyprocess": ["ptyprocess"],
    "pyparsing": ["pyparsing"],
    "referencing": ["referencing"],
    "rpds-py": ["rpds"],
    "setproctitle": ["setproctitle"],
    "six": ["six"],
    "soupsieve": ["soupsieve"],
    "tinycss": ["tinycss"],
    "tinycss2": ["tinycss2"],
    "tzdata": ["tzdata"],
    "urllib3": ["urllib3"],
    "webencodings": ["webencodings"],
    "setuptools": ["setuptools", "pkg_resources"],
    "wheel": ["wheel"],
    "charset-normalizer": ["charset_normalizer"],
    "fasteners": ["fasteners"],
    "iniconfig": ["iniconfig"],
    "jsonschema": ["jsonschema"],
    "jsonschema-specifications": ["jsonschema_specifications"],
    "llvmlite": ["llvmlite"],
    "Mako": ["mako"],
    "MarkupSafe": ["markupsafe"],
    "mccabe": ["mccabe"],
    "monotonic": ["monotonic"],
    "mypy-extensions": ["mypy_extensions"],
    "mypy_extensions": ["mypy_extensions"],
    "narwhals": ["narwhals"],
}

def get_installed_packages(requirements_path):
    packages = []
    with open(requirements_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Remove version specifiers
            pkg_name = re.split(r'[<>=!]', line)[0].strip()
            if pkg_name:
                packages.append(pkg_name)
    return packages

def find_imports_in_file(filepath):
    imports = set()
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Regex for "import X"
            matches_import = re.findall(r'^\s*import\s+([\w\.]+)', content, re.MULTILINE)
            for m in matches_import:
                imports.add(m.split('.')[0])
            
            # Regex for "from X import Y"
            matches_from = re.findall(r'^\s*from\s+([\w\.]+)\s+import', content, re.MULTILINE)
            for m in matches_from:
                imports.add(m.split('.')[0])
    except Exception as e:
        pass
    return imports

def get_all_imports(root_dir):
    all_imports = set()
    py_files = []
    # Directories to completely ignore
    IGNORED_DIRS = {'venv', 'env', 'site-packages', 'vendor', 'scripts', 'Hidrogenio'}
    
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden and env dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in IGNORED_DIRS]
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                py_files.append(full_path)
                file_imports = find_imports_in_file(full_path)
                all_imports.update(file_imports)
    return all_imports, py_files

def run_grep(pkg_name, import_names, root_dir):
    # Search for the package name AND the import names
    search_terms = set([pkg_name] + import_names)
    results = {}
    
    # Exclude directories arguments
    exclude_args = []
    for d in ['scripts', 'vendor', 'Hidrogenio', 'site-packages', 'venv']:
        exclude_args.extend(['--exclude-dir', d])
    
    for term in search_terms:
        # grep -r "term" . --exclude-dir=... --include="*.py" | head -n 3
        try:
            cmd = ['grep', '-r', term, root_dir] + exclude_args + ['--include=*.py', '-l']
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout.strip().split('\n')
            output = [line for line in output if line]
            if output:
                results[term] = output
        except Exception as e:
            results[term] = [f"Error: {e}"]
    return results

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir) # Parent of scripts
    req_file = os.path.join(root_dir, 'requirements.txt')
    
    print(f"Analyzing requirements from: {req_file}")
    packages = get_installed_packages(req_file)
    print(f"Found {len(packages)} packages.")
    
    print(f"Scanning usage in: {root_dir}")
    used_imports, py_files = get_all_imports(root_dir)
    print(f"Found {len(py_files)} Python files.")
    print(f"Detected {len(used_imports)} unique top-level imports.")
    
    print("\n" + "="*60)
    print("ANALYSIS RESULT")
    print("="*60)
    
    unused_count = 0
    
    for pkg in sorted(packages):
        possible_imports = PACKAGE_TO_IMPORTS.get(pkg, [pkg, pkg.replace('-', '_')])
        
        is_used = False
        for imp in possible_imports:
            if imp in used_imports:
                is_used = True
                break
        
        if is_used:
            print(f"[USED] {pkg}")
        else:
            # If not found in imports, do a grep check to be sure (maybe used in string/comment/subprocess or missed)
            # The user specifically asked for "detailed tracking run with grep -r"
            grep_hits = run_grep(pkg, possible_imports, root_dir)
            
            if grep_hits:
                print(f"[MAYBE USED?] {pkg} - Not in imports, but grep found matches:")
                for term, files in grep_hits.items():
                    print(f"  - Term '{term}' found in {len(files)} files (e.g. {files[:2]})")
            else:
                print(f"[UNUSED] {pkg} - No imports or grep matches found.")
                unused_count += 1
                
    print("\n" + "="*60)
    print(f"Total Unused (Confident): {unused_count}")

if __name__ == "__main__":
    main()
