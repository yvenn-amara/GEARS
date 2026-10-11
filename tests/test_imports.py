import subprocess
import sys

def test_lightweight_import():
    """
    Verify that 'import gears' does not trigger eager loading of heavy dependencies.
    We run this in a subprocess to isolate the import and check for side effects.
    """
    # List of modules that should NOT be loaded during a bare 'import gears'
    heavy_libs = [
        "torch",
        "neuralforecast",
        "pmdarima",
        "statsmodels",
        "matplotlib",
        "requests",
    ]
    
    # Python code to run in the subprocess
    code = """
import sys
try:
    import gears
    import pprint
    pprint.pprint(sys.modules.keys())
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""
    
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Subprocess failed: {stderr}")
        sys.exit(1)
    
    # Check if any of the heavy libs are in the loaded modules list
    for lib in heavy_libs:
        # We look for the module name as a key in the printed dict/list
        if f"'{lib}'" in stdout or f'"{lib}"' in stdout:
            print(f"FAILURE: Heavy library '{lib}' was eagerly loaded during 'import gears'!")
            sys.exit(1)

if __name__ == "__main__":
    try:
        test_lightweight_import()
        print("Lightweight import test passed!")
    except Exception as e:
        print(f"Lightweight import test failed: {e}")
        sys.exit(1)
