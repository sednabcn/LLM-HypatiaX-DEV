# validate_paper_config.py
import os

required_env = {
    "PYSR_TIMEOUT": "120",
    "METHOD_TIMEOUT": "360", 
    "POPULATIONS": "30",
    "N_ITERATIONS": "1000",
    "NN_SEED": "42",
    "PYSR_SEED": "42",
}

print("=" * 68)
print("PAPER CONFIGURATION VALIDATION")
print("=" * 68)

all_ok = True
for var, expected in required_env.items():
    actual = os.environ.get(var)
    if actual == expected:
        print(f"  ✓ {var}={actual}")
    elif actual is None:
        print(f"  ✗ {var} not set (expected {expected})")
        all_ok = False
    else:
        print(f"  ⚠ {var}={actual} (expected {expected})")
        all_ok = False

print("=" * 68)
if all_ok:
    print("✓ Paper configuration is CORRECT")
    print("  Results will match the paper's targets")
else:
    print("✗ Paper configuration is INCORRECT")
    print("  Set environment variables before running:")
    for var, val in required_env.items():
        print(f"    export {var}={val}")
print("=" * 68)
