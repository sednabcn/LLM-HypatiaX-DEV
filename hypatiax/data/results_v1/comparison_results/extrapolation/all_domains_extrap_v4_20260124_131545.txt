──(py312)(agagora㉿localhost)-[~/Downloads/GITHUB/LLM-HypatiaX-Colab/experiments/pyfigures/5_data]
└─$ python standalone_real_methods_test.py --all --extrapolation

================================================================================
        STANDALONE TEST SUITE v4 - ALL DOMAINS + EXTRAPOLATION (FIXED!)
================================================================================

📦 Loading real method implementations...
✅ Pure LLM loaded
✅ Neural Network loaded
/home/agagora/Downloads/py312/lib/python3.12/site-packages/juliacall/__init__.py:61: UserWarning: torch was imported before juliacall. This may cause a segfault. To avoid this, import juliacall before importing torch. For updates, see https://github.com/pytorch/pytorch/issues/78829.
  warnings.warn(
✅ Hybrid System v40 loaded (with sanitization)

✅ Loaded 3 methods
   • Pure LLM
   • Neural Network
   • Hybrid System v40

🔬 Extrapolation testing: ENABLED (v4 FIXES APPLIED)
   Regimes: Near (1.2×), Medium (2×), Far (5×)
   Enhanced debugging and error handling enabled
================================================================================


📊 Running 15 tests across 5 domains


================================================================================
                             DOMAIN 1/5: CHEMISTRY
================================================================================

[1/15] arrhenius
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:00:19,409 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.9996 ✓
   Running Hybrid System v40... 2026-01-24 13:00:41,573 - INFO - ======================================================================
2026-01-24 13:00:41,574 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:00:41,574 - INFO - ======================================================================
2026-01-24 13:00:41,574 - INFO - Domain: chemistry
2026-01-24 13:00:41,574 - INFO - Discovery mode: calibrated
2026-01-24 13:00:41,575 - INFO - Primary LLM: anthropic
2026-01-24 13:00:41,575 - INFO - Auto-config: True
2026-01-24 13:00:41,575 - INFO - Max retries: 5
2026-01-24 13:00:41,575 - INFO - Variable sanitization: ENABLED
2026-01-24 13:00:41,575 - INFO - ======================================================================
2026-01-24 13:00:41,576 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:01:12,261 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Arrhenius Equation: k = A*exp(-Ea/(R*T))
Domain: CHEMISTRY
Samples: 200
Variables: ['T']
Equation hint: arrhenius
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:01:12,262 - INFO -    Sanitized: T → var_T
2026-01-24 13:01:12,262 - INFO -
[SANITIZATION] Detected problematic variable names
2026-01-24 13:01:12,262 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: var_T
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: (0.21613532 - (((var_T * var_T) * 8.6731875e-9) * var_T)) * (-0.22207917 - (((var_T * var_T) * 3.2220527e-8) * (((var_T * 8.562808e-5) + -0.02861313) * (var_T * var_T))))
   R²: 0.9816
2026-01-24 13:02:42,852 - INFO -    Result: (0.21613532 - (((var_T * var_T) * 8.6731875e-9) * var_T)) * (-0.22207917 - (((var_T * var_T) * 3.2220527e-8) * (((var_T * 8.562808e-5) + -0.02861313) * (var_T * var_T))))
2026-01-24 13:02:42,852 - INFO -    R² = 0.9816
2026-01-24 13:02:42,853 - INFO -    Restored: (0.21613532 - (((T * T) * 8.6731875e-9) * T)) * (-0.22207917 - (((T * T) * 3.2220527e-8) * (((T * 8.562808e-5) + -0.02861313) * (T * T))))
2026-01-24 13:02:42,854 - WARNING -    [WARNING] Possible overfit
2026-01-24 13:02:42,854 - WARNING -       High complexity (170) but R²=0.9816
2026-01-24 13:02:42,854 - WARNING -       Many constants detected (6)
2026-01-24 13:02:42,854 - INFO -    [BEST] New best!
2026-01-24 13:02:42,855 - INFO -
[SYMBOLIC] Attempt 2/5 (seed=43)

[DISCOVERY] Starting symbolic regression...
   Variables: var_T
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: ((((var_T * -0.0025503593) + 1.2996379) * var_T) / (var_T + -427.0217)) - (((var_T * -0.002878961) + 1.2719337) * (var_T / (var_T + -397.0573)))
   R²: 0.9989
2026-01-24 13:03:07,402 - INFO -    Result: ((((var_T * -0.0025503593) + 1.2996379) * var_T) / (var_T + -427.0217)) - (((var_T * -0.002878961) + 1.2719337) * (var_T / (var_T + -397.0573)))
2026-01-24 13:03:07,403 - INFO -    R² = 0.9989
2026-01-24 13:03:07,403 - INFO -    Restored: ((((T * -0.0025503593) + 1.2996379) * T) / (T + -427.0217)) - (((T * -0.002878961) + 1.2719337) * (T / (T + -397.0573)))
2026-01-24 13:03:07,403 - WARNING -    [WARNING] Possible overfit
2026-01-24 13:03:07,404 - WARNING -       High complexity (144) but R²=0.9989
2026-01-24 13:03:07,404 - WARNING -       Many constants detected (6)
2026-01-24 13:03:07,404 - INFO -    [BEST] New best!
2026-01-24 13:03:07,404 - INFO -
[SYMBOLIC] Attempt 3/5 (seed=44)

[DISCOVERY] Starting symbolic regression...
   Variables: var_T
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: ((((var_T + -2.29927) * 1.0074962) - var_T) * (((var_T + -1.8135029) * 1.0055704) - var_T)) * ((((var_T * 1.00646) - var_T) + -1.8129953) * 7.531144)
   R²: 0.9880
2026-01-24 13:03:33,690 - INFO -    Result: ((((var_T + -2.29927) * 1.0074962) - var_T) * (((var_T + -1.8135029) * 1.0055704) - var_T)) * ((((var_T * 1.00646) - var_T) + -1.8129953) * 7.531144)
2026-01-24 13:03:33,691 - INFO -    R² = 0.9880
2026-01-24 13:03:33,691 - INFO -    Restored: ((((T + -2.29927) * 1.0074962) - T) * (((T + -1.8135029) * 1.0055704) - T)) * ((((T * 1.00646) - T) + -1.8129953) * 7.531144)
2026-01-24 13:03:33,691 - WARNING -    [WARNING] Possible overfit
2026-01-24 13:03:33,692 - WARNING -       High complexity (149) but R²=0.9880
2026-01-24 13:03:33,692 - WARNING -       Many constants detected (7)
2026-01-24 13:03:33,692 - INFO -
[SYMBOLIC] Attempt 4/5 (seed=45)

[DISCOVERY] Starting symbolic regression...
   Variables: var_T
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: ((11.902848 / (386.0505 - var_T)) + (135.88815 / var_T)) - 0.5959763
   R²: 0.9972
2026-01-24 13:03:56,732 - INFO -    Result: ((11.902848 / (386.0505 - var_T)) + (135.88815 / var_T)) - 0.5959763
2026-01-24 13:03:56,732 - INFO -    R² = 0.9972
2026-01-24 13:03:56,733 - INFO -    Restored: ((11.902848 / (386.0505 - T)) + (135.88815 / T)) - 0.5959763
2026-01-24 13:03:56,733 - WARNING -    [WARNING] Possible overfit
2026-01-24 13:03:56,733 - WARNING -       High complexity (68) but R²=0.9972
2026-01-24 13:03:56,734 - INFO -
[SYMBOLIC] Attempt 5/5 (seed=46)

[DISCOVERY] Starting symbolic regression...
   Variables: var_T
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: ((var_T * ((((var_T / 0.8649331) * var_T) * 1.7816167e-7) + -0.057993587)) - -11.77187) * ((var_T * ((((var_T / 0.8649331) * var_T) * 1.7816167e-7) + -0.057993587)) - -11.77187)
   R²: 0.9814
2026-01-24 13:04:22,044 - INFO -    Result: ((var_T * ((((var_T / 0.8649331) * var_T) * 1.7816167e-7) + -0.057993587)) - -11.77187) * ((var_T * ((((var_T / 0.8649331) * var_T) * 1.7816167e-7) + -0.057993587)) - -11.77187)
2026-01-24 13:04:22,044 - INFO -    R² = 0.9814
2026-01-24 13:04:22,044 - INFO -    Restored: ((T * ((((T / 0.8649331) * T) * 1.7816167e-7) + -0.057993587)) - -11.77187) * ((T * ((((T / 0.8649331) * T) * 1.7816167e-7) + -0.057993587)) - -11.77187)
2026-01-24 13:04:22,045 - WARNING -    [WARNING] Possible overfit
2026-01-24 13:04:22,045 - WARNING -       High complexity (177) but R²=0.9814
2026-01-24 13:04:22,045 - WARNING -       Many constants detected (8)
2026-01-24 13:04:22,046 - INFO -
[SUCCESS] SymbolicEngine succeeded (R²=0.9989)

[OK] Discovery complete
   Expression: ((((T * -0.0025503593) + 1.2996379) * T) / (T + -427.0217)) - (((T * -0.002878961) + 1.2719337) * (T / (T + -397.0573)))
   R² Score: 0.9989
   Engine: symbolic
   Attempt: 2/5
   Variables sanitized: ['T']

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 70.0/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=0.9989, Extrap: 0.0% ✓

[2/15] henderson_hasselbalch
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:04:30,534 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.9989, Extrap: 9.5% ✓
   Running Hybrid System v40... 2026-01-24 13:04:31,751 - INFO - ======================================================================
2026-01-24 13:04:31,751 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:04:31,751 - INFO - ======================================================================
2026-01-24 13:04:31,752 - INFO - Domain: chemistry
2026-01-24 13:04:31,752 - INFO - Discovery mode: calibrated
2026-01-24 13:04:31,752 - INFO - Primary LLM: anthropic
2026-01-24 13:04:31,752 - INFO - Auto-config: True
2026-01-24 13:04:31,752 - INFO - Max retries: 5
2026-01-24 13:04:31,753 - INFO - Variable sanitization: ENABLED
2026-01-24 13:04:31,753 - INFO - ======================================================================
2026-01-24 13:04:31,753 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:04:32,692 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Henderson-Hasselbalch: pH = pKa + log10([A-]/[HA])
Domain: CHEMISTRY
Samples: 200
Variables: ['A_minus', 'HA']
Equation hint: henderson_hasselbalch
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:04:32,692 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: A_minus, HA
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: (((A_minus / A_minus) / (HA + 0.5249313)) + ((A_minus + 36.366962) * 0.16462906)) + (-0.80636704 / ((A_minus + A_minus) - -0.4938591))
   R²: 0.9991
2026-01-24 13:04:56,636 - INFO -    Result: (((A_minus / A_minus) / (HA + 0.5249313)) + ((A_minus + 36.366962) * 0.16462906)) + (-0.80636704 / ((A_minus + A_minus) - -0.4938591))
2026-01-24 13:04:56,636 - INFO -    R² = 0.9991
2026-01-24 13:04:56,637 - INFO -    [BEST] New best!
2026-01-24 13:04:56,637 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: (((A_minus / A_minus) / (HA + 0.5249313)) + ((A_minus + 36.366962) * 0.16462906)) + (-0.80636704 / ((A_minus + A_minus) - -0.4938591))
   R² Score: 0.9991
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 70.0/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=0.9991, Extrap: 0.0% ✓

[3/15] rate_law
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:05:04,621 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.9994, Extrap: 2720.0% ✓
   Running Hybrid System v40... 2026-01-24 13:05:05,664 - INFO - ======================================================================
2026-01-24 13:05:05,665 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:05:05,666 - INFO - ======================================================================
2026-01-24 13:05:05,666 - INFO - Domain: chemistry
2026-01-24 13:05:05,666 - INFO - Discovery mode: calibrated
2026-01-24 13:05:05,667 - INFO - Primary LLM: anthropic
2026-01-24 13:05:05,667 - INFO - Auto-config: True
2026-01-24 13:05:05,668 - INFO - Max retries: 5
2026-01-24 13:05:05,668 - INFO - Variable sanitization: ENABLED
2026-01-24 13:05:05,669 - INFO - ======================================================================
2026-01-24 13:05:05,669 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:05:06,573 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Rate Law: rate = k*[A]²*[B]
Domain: CHEMISTRY
Samples: 200
Variables: ['A_conc', 'B_conc']
Equation hint: rate_law
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:05:06,574 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: A_conc, B_conc
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: (((A_conc * A_conc) * -4.333499) * B_conc) * -0.115380205
   R²: 1.0000
2026-01-24 13:05:31,003 - INFO -    Result: (((A_conc * A_conc) * -4.333499) * B_conc) * -0.115380205
2026-01-24 13:05:31,004 - INFO -    R² = 1.0000
2026-01-24 13:05:31,004 - INFO -    [BEST] New best!
2026-01-24 13:05:31,004 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: (((A_conc * A_conc) * -4.333499) * B_conc) * -0.115380205
   R² Score: 1.0000
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 94.9/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=1.0000, Extrap: 0.0% ✓

────────────────────────────────────────────────────────────────────────────────
DOMAIN SUMMARY: CHEMISTRY
────────────────────────────────────────────────────────────────────────────────
  Pure LLM                       Success: 3/3 (100.0%), Avg R²: 1.0000
  Neural Network                 Success: 3/3 (100.0%), Avg R²: 0.9993, Avg Extrap: 1364.7%
  Hybrid System v40              Success: 3/3 (100.0%), Avg R²: 0.9993, Avg Extrap:    0.0%

================================================================================
                              DOMAIN 2/5: BIOLOGY
================================================================================

[4/15] allometric_scaling
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:05:38,629 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=1.0000, Extrap: 66.5% ✓
   Running Hybrid System v40... 2026-01-24 13:05:40,350 - INFO - ======================================================================
2026-01-24 13:05:40,350 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:05:40,350 - INFO - ======================================================================
2026-01-24 13:05:40,350 - INFO - Domain: biology
2026-01-24 13:05:40,351 - INFO - Discovery mode: calibrated
2026-01-24 13:05:40,351 - INFO - Primary LLM: anthropic
2026-01-24 13:05:40,351 - INFO - Auto-config: True
2026-01-24 13:05:40,351 - INFO - Max retries: 5
2026-01-24 13:05:40,351 - INFO - Variable sanitization: ENABLED
2026-01-24 13:05:40,352 - INFO - ======================================================================
2026-01-24 13:05:40,352 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:05:41,901 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Allometric Scaling: Y = a*M^b (metabolic rate)
Domain: BIOLOGY
Samples: 200
Variables: ['M']
Equation hint: allometric_scaling
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:05:41,901 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: M
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: ((M - -1.8721828) - ((M * -22.740671) / (M - -18.916243))) + (M * (M * -0.0010350774))
   R²: 1.0000
2026-01-24 13:06:07,860 - INFO -    Result: ((M - -1.8721828) - ((M * -22.740671) / (M - -18.916243))) + (M * (M * -0.0010350774))
2026-01-24 13:06:07,860 - INFO -    R² = 1.0000
2026-01-24 13:06:07,861 - INFO -    [BEST] New best!
2026-01-24 13:06:07,861 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: ((M - -1.8721828) - ((M * -22.740671) / (M - -18.916243))) + (M * (M * -0.0010350774))
   R² Score: 1.0000
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 77.8/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=1.0000, Extrap: 0.0% ✓

[5/15] michaelis_menten
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:06:14,955 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.9999, Extrap: 96.4% ✓
   Running Hybrid System v40... 2026-01-24 13:06:15,969 - INFO - ======================================================================
2026-01-24 13:06:15,969 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:06:15,969 - INFO - ======================================================================
2026-01-24 13:06:15,969 - INFO - Domain: biology
2026-01-24 13:06:15,970 - INFO - Discovery mode: calibrated
2026-01-24 13:06:15,970 - INFO - Primary LLM: anthropic
2026-01-24 13:06:15,970 - INFO - Auto-config: True
2026-01-24 13:06:15,970 - INFO - Max retries: 5
2026-01-24 13:06:15,970 - INFO - Variable sanitization: ENABLED
2026-01-24 13:06:15,970 - INFO - ======================================================================
2026-01-24 13:06:15,971 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:06:16,872 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Michaelis-Menten: v = (Vmax*[S])/(Km+[S])
Domain: BIOLOGY
Samples: 200
Variables: ['var_S']
Equation hint: michaelis_menten
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:06:16,873 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: var_S
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: (var_S * 50.000072) / (var_S + (10.000014 + (var_S * 1.4730435e-6)))
   R²: 1.0000
2026-01-24 13:06:40,913 - INFO -    Result: (var_S * 50.000072) / (var_S + (10.000014 + (var_S * 1.4730435e-6)))
2026-01-24 13:06:40,913 - INFO -    R² = 1.0000
2026-01-24 13:06:40,914 - INFO -    [BEST] New best!
2026-01-24 13:06:40,914 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: (var_S * 50.000072) / (var_S + (10.000014 + (var_S * 1.4730435e-6)))
   R² Score: 1.0000
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 82.9/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=1.0000, Extrap: 0.0% ✓

[6/15] logistic_growth
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:06:47,871 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=1.0000, Extrap: 6818.5% ✓
   Running Hybrid System v40... 2026-01-24 13:06:48,899 - INFO - ======================================================================
2026-01-24 13:06:48,899 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:06:48,899 - INFO - ======================================================================
2026-01-24 13:06:48,899 - INFO - Domain: biology
2026-01-24 13:06:48,900 - INFO - Discovery mode: calibrated
2026-01-24 13:06:48,900 - INFO - Primary LLM: anthropic
2026-01-24 13:06:48,900 - INFO - Auto-config: True
2026-01-24 13:06:48,900 - INFO - Max retries: 5
2026-01-24 13:06:48,900 - INFO - Variable sanitization: ENABLED
2026-01-24 13:06:48,900 - INFO - ======================================================================
2026-01-24 13:06:48,901 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:06:49,811 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Logistic Growth: dN/dt = r*N*(1-N/K)
Domain: BIOLOGY
Samples: 200
Variables: ['var_N']
Equation hint: logistic_growth
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:06:49,812 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: var_N
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: ((-0.0003 * var_N) - -0.3) * var_N
   R²: 1.0000
2026-01-24 13:07:14,662 - INFO -    Result: ((-0.0003 * var_N) - -0.3) * var_N
2026-01-24 13:07:14,662 - INFO -    R² = 1.0000
2026-01-24 13:07:14,662 - INFO -    [BEST] New best!
2026-01-24 13:07:14,663 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: ((-0.0003 * var_N) - -0.3) * var_N
   R² Score: 1.0000
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 95.5/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=1.0000, Extrap: 0.0% ✓

────────────────────────────────────────────────────────────────────────────────
DOMAIN SUMMARY: BIOLOGY
────────────────────────────────────────────────────────────────────────────────
  Pure LLM                       Success: 3/3 (100.0%), Avg R²: 1.0000
  Neural Network                 Success: 3/3 (100.0%), Avg R²: 0.9999, Avg Extrap: 2327.1%
  Hybrid System v40              Success: 3/3 (100.0%), Avg R²: 1.0000, Avg Extrap:    0.0%

================================================================================
                              DOMAIN 3/5: PHYSICS
================================================================================

[7/15] kinetic_energy
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:07:21,338 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.9997, Extrap: 4804.2% ✓
   Running Hybrid System v40... 2026-01-24 13:07:22,378 - INFO - ======================================================================
2026-01-24 13:07:22,379 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:07:22,379 - INFO - ======================================================================
2026-01-24 13:07:22,380 - INFO - Domain: physics
2026-01-24 13:07:22,380 - INFO - Discovery mode: calibrated
2026-01-24 13:07:22,381 - INFO - Primary LLM: anthropic
2026-01-24 13:07:22,381 - INFO - Auto-config: True
2026-01-24 13:07:22,382 - INFO - Max retries: 5
2026-01-24 13:07:22,382 - INFO - Variable sanitization: ENABLED
2026-01-24 13:07:22,383 - INFO - ======================================================================
2026-01-24 13:07:22,383 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:07:23,289 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Kinetic Energy: KE = 0.5*m*v²
Domain: PHYSICS
Samples: 200
Variables: ['m', 'v']
Equation hint: kinetic_energy
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:07:23,290 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: m, v
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: v * (m * (v * 0.5))
   R²: 1.0000
2026-01-24 13:07:49,415 - INFO -    Result: v * (m * (v * 0.5))
2026-01-24 13:07:49,416 - INFO -    R² = 1.0000
2026-01-24 13:07:49,416 - INFO -    [BEST] New best!
2026-01-24 13:07:49,417 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: v * (m * (v * 0.5))
   R² Score: 1.0000
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 95.5/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=1.0000, Extrap: 0.0% ✓

[8/15] gravitational_force
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:07:55,393 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.2932 ✓
   Running Hybrid System v40... 2026-01-24 13:07:56,438 - INFO - ======================================================================
2026-01-24 13:07:56,439 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:07:56,439 - INFO - ======================================================================
2026-01-24 13:07:56,440 - INFO - Domain: physics
2026-01-24 13:07:56,440 - INFO - Discovery mode: calibrated
2026-01-24 13:07:56,441 - INFO - Primary LLM: anthropic
2026-01-24 13:07:56,441 - INFO - Auto-config: True
2026-01-24 13:07:56,442 - INFO - Max retries: 5
2026-01-24 13:07:56,442 - INFO - Variable sanitization: ENABLED
2026-01-24 13:07:56,443 - INFO - ======================================================================
2026-01-24 13:07:56,443 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:07:57,355 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Gravitational Force: F = G*m1*m2/r²
Domain: PHYSICS
Samples: 200
Variables: ['m1', 'm2', 'r']
Equation hint: gravitational_force
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:07:57,355 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: m1, m2, r
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: -0.22981945
   R²: -0.0257
2026-01-24 13:08:20,279 - INFO -    Result: -0.22981945
2026-01-24 13:08:20,279 - INFO -    R² = -0.0257
2026-01-24 13:08:20,280 - INFO -    [BEST] New best!
2026-01-24 13:08:20,280 - INFO -
[SYMBOLIC] Attempt 2/5 (seed=43)

[DISCOVERY] Starting symbolic regression...
   Variables: m1, m2, r
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
    ✅ Found: -0.5971072
   R²: -0.0257
2026-01-24 13:08:43,133 - INFO -    Result: -0.5971072
2026-01-24 13:08:43,134 - INFO -    R² = -0.0257
2026-01-24 13:08:43,134 - INFO -
[SYMBOLIC] Attempt 3/5 (seed=44)

[DISCOVERY] Starting symbolic regression...
   Variables: m1, m2, r
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: 1.4503475
   R²: -0.0257
2026-01-24 13:09:05,962 - INFO -    Result: 1.4503475
2026-01-24 13:09:05,963 - INFO -    R² = -0.0257
2026-01-24 13:09:05,963 - INFO -
[SYMBOLIC] Attempt 4/5 (seed=45)

[DISCOVERY] Starting symbolic regression...
   Variables: m1, m2, r
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: 1.3109097
   R²: -0.0257
2026-01-24 13:09:28,309 - INFO -    Result: 1.3109097
2026-01-24 13:09:28,310 - INFO -    R² = -0.0257
2026-01-24 13:09:28,310 - INFO -
[SYMBOLIC] Attempt 5/5 (seed=46)

[DISCOVERY] Starting symbolic regression...
   Variables: m1, m2, r
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: 0.6298781
   R²: -0.0257
2026-01-24 13:09:49,684 - INFO -    Result: 0.6298781
2026-01-24 13:09:49,684 - INFO -    R² = -0.0257
2026-01-24 13:09:49,684 - WARNING -
[WARNING] SymbolicEngine best R²=-0.0257

[OK] Discovery complete
   Expression: -0.22981945
   R² Score: -0.0257
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 100.0/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================

✗ Failed

[9/15] ideal_gas_law
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:09:55,501 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.8136, Extrap: 1.8% ✓
   Running Hybrid System v40... 2026-01-24 13:09:56,538 - INFO - ======================================================================
2026-01-24 13:09:56,538 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:09:56,538 - INFO - ======================================================================
2026-01-24 13:09:56,538 - INFO - Domain: physics
2026-01-24 13:09:56,539 - INFO - Discovery mode: calibrated
2026-01-24 13:09:56,539 - INFO - Primary LLM: anthropic
2026-01-24 13:09:56,539 - INFO - Auto-config: True
2026-01-24 13:09:56,539 - INFO - Max retries: 5
2026-01-24 13:09:56,539 - INFO - Variable sanitization: ENABLED
2026-01-24 13:09:56,540 - INFO - ======================================================================
2026-01-24 13:09:56,540 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:09:57,455 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Ideal Gas Law: P = nRT/V
Domain: PHYSICS
Samples: 200
Variables: ['n', 'T', 'V']
Equation hint: ideal_gas_law
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:09:57,456 - INFO -    Sanitized: T → var_T
2026-01-24 13:09:57,456 - INFO -    Sanitized: V → var_V
2026-01-24 13:09:57,456 - INFO -
[SANITIZATION] Detected problematic variable names
2026-01-24 13:09:57,457 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: n, var_T, var_V
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: ((8.313999 / var_V) * (var_T - -2.0213603e-5)) * n
   R²: 1.0000
2026-01-24 13:10:23,696 - INFO -    Result: ((8.313999 / var_V) * (var_T - -2.0213603e-5)) * n
2026-01-24 13:10:23,697 - INFO -    R² = 1.0000
2026-01-24 13:10:23,697 - INFO -    Restored: ((8.313999 / V) * (T - -2.0213603e-5)) * n
2026-01-24 13:10:23,698 - INFO -    [BEST] New best!
2026-01-24 13:10:23,698 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: ((8.313999 / V) * (T - -2.0213603e-5)) * n
   R² Score: 1.0000
   Engine: symbolic
   Attempt: 1/5
   Variables sanitized: ['T', 'V']

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 86.5/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=1.0000, Extrap: 0.0% ✓

────────────────────────────────────────────────────────────────────────────────
DOMAIN SUMMARY: PHYSICS
────────────────────────────────────────────────────────────────────────────────
  Pure LLM                       Success: 3/3 (100.0%), Avg R²: 1.0000
  Neural Network                 Success: 3/3 (100.0%), Avg R²: 0.7022, Avg Extrap: 2403.0%
  Hybrid System v40              Success: 2/3 ( 66.7%), Avg R²: 1.0000, Avg Extrap:    0.0%

================================================================================
                              DOMAIN 4/5: DEFI_AMM
================================================================================

[10/15] impermanent_loss
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:10:30,752 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.9997, Extrap: 83.6% ✓
   Running Hybrid System v40... 2026-01-24 13:10:31,907 - INFO - ======================================================================
2026-01-24 13:10:31,907 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:10:31,908 - INFO - ======================================================================
2026-01-24 13:10:31,908 - INFO - Domain: defi_amm
2026-01-24 13:10:31,908 - INFO - Discovery mode: calibrated
2026-01-24 13:10:31,908 - INFO - Primary LLM: anthropic
2026-01-24 13:10:31,908 - INFO - Auto-config: True
2026-01-24 13:10:31,909 - INFO - Max retries: 5
2026-01-24 13:10:31,909 - INFO - Variable sanitization: ENABLED
2026-01-24 13:10:31,909 - INFO - ======================================================================
2026-01-24 13:10:31,909 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:10:32,835 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Impermanent Loss: IL = 2*sqrt(r)/(1+r) - 1
Domain: DEFI_AMM
Samples: 200
Variables: ['price_ratio']
Equation hint: impermanent_loss
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:10:32,836 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: price_ratio
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: ((price_ratio + ((price_ratio * -1.0872616) + 0.17365843)) - (0.0865178 / price_ratio)) * (((price_ratio * -1.1635232) + (1.6808376 - (0.12216556 / price_ratio))) + price_ratio)
   R²: 1.0000
2026-01-24 13:10:56,036 - INFO -    Result: ((price_ratio + ((price_ratio * -1.0872616) + 0.17365843)) - (0.0865178 / price_ratio)) * (((price_ratio * -1.1635232) + (1.6808376 - (0.12216556 / price_ratio))) + price_ratio)
2026-01-24 13:10:56,037 - INFO -    R² = 1.0000
2026-01-24 13:10:56,037 - INFO -    [BEST] New best!
2026-01-24 13:10:56,037 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: ((price_ratio + ((price_ratio * -1.0872616) + 0.17365843)) - (0.0865178 / price_ratio)) * (((price_ratio * -1.1635232) + (1.6808376 - (0.12216556 / price_ratio))) + price_ratio)
   R² Score: 1.0000
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 94.0/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=1.0000, Extrap: 0.0% ✓

[11/15] price_impact
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:11:03,486 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.9991, Extrap: 44.4% ✓
   Running Hybrid System v40... 2026-01-24 13:11:04,527 - INFO - ======================================================================
2026-01-24 13:11:04,528 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:11:04,528 - INFO - ======================================================================
2026-01-24 13:11:04,528 - INFO - Domain: defi_amm
2026-01-24 13:11:04,528 - INFO - Discovery mode: calibrated
2026-01-24 13:11:04,528 - INFO - Primary LLM: anthropic
2026-01-24 13:11:04,529 - INFO - Auto-config: True
2026-01-24 13:11:04,529 - INFO - Max retries: 5
2026-01-24 13:11:04,529 - INFO - Variable sanitization: ENABLED
2026-01-24 13:11:04,529 - INFO - ======================================================================
2026-01-24 13:11:04,529 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:11:05,456 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Price Impact: impact = dx/(x+dx)
Domain: DEFI_AMM
Samples: 200
Variables: ['reserve', 'swap']
Equation hint: price_impact
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:11:05,457 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: reserve, swap
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: swap / (swap + reserve)
   R²: 1.0000
2026-01-24 13:11:25,035 - INFO -    Result: swap / (swap + reserve)
2026-01-24 13:11:25,035 - INFO -    R² = 1.0000
2026-01-24 13:11:25,036 - INFO -    [BEST] New best!
2026-01-24 13:11:25,036 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: swap / (swap + reserve)
   R² Score: 1.0000
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 83.5/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=1.0000, Extrap: 0.0% ✓

[12/15] constant_product
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:11:31,952 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.9967, Extrap: 6.8% ✓
   Running Hybrid System v40... 2026-01-24 13:11:32,977 - INFO - ======================================================================
2026-01-24 13:11:32,977 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:11:32,977 - INFO - ======================================================================
2026-01-24 13:11:32,978 - INFO - Domain: defi_amm
2026-01-24 13:11:32,978 - INFO - Discovery mode: calibrated
2026-01-24 13:11:32,978 - INFO - Primary LLM: anthropic
2026-01-24 13:11:32,978 - INFO - Auto-config: True
2026-01-24 13:11:32,978 - INFO - Max retries: 5
2026-01-24 13:11:32,978 - INFO - Variable sanitization: ENABLED
2026-01-24 13:11:32,979 - INFO - ======================================================================
2026-01-24 13:11:32,979 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:11:33,890 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Constant Product: y = k/x
Domain: DEFI_AMM
Samples: 200
Variables: ['x']
Equation hint: constant_product
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:11:33,891 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: x
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: 1038.933 / (x / 962.526)
   R²: 1.0000
2026-01-24 13:11:56,610 - INFO -    Result: 1038.933 / (x / 962.526)
2026-01-24 13:11:56,611 - INFO -    R² = 1.0000
2026-01-24 13:11:56,611 - INFO -    [BEST] New best!
2026-01-24 13:11:56,611 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: 1038.933 / (x / 962.526)
   R² Score: 1.0000
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 94.6/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=1.0000, Extrap: 0.0% ✓

────────────────────────────────────────────────────────────────────────────────
DOMAIN SUMMARY: DEFI_AMM
────────────────────────────────────────────────────────────────────────────────
  Pure LLM                       Success: 3/3 (100.0%), Avg R²: 1.0000
  Neural Network                 Success: 3/3 (100.0%), Avg R²: 0.9985, Avg Extrap:   44.9%
  Hybrid System v40              Success: 3/3 (100.0%), Avg R²: 1.0000, Avg Extrap:    0.0%

================================================================================
                             DOMAIN 5/5: DEFI_RISK
================================================================================

[13/15] var_95
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:12:03,656 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.9998, Extrap: 1139.7% ✓
   Running Hybrid System v40... 2026-01-24 13:12:04,773 - INFO - ======================================================================
2026-01-24 13:12:04,774 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:12:04,774 - INFO - ======================================================================
2026-01-24 13:12:04,774 - INFO - Domain: defi_risk
2026-01-24 13:12:04,774 - INFO - Discovery mode: calibrated
2026-01-24 13:12:04,775 - INFO - Primary LLM: anthropic
2026-01-24 13:12:04,775 - INFO - Auto-config: True
2026-01-24 13:12:04,775 - INFO - Max retries: 5
2026-01-24 13:12:04,775 - INFO - Variable sanitization: ENABLED
2026-01-24 13:12:04,775 - INFO - ======================================================================
2026-01-24 13:12:04,776 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:12:05,682 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Value at Risk 95%: VaR = P*σ*1.645
Domain: DEFI_RISK
Samples: 200
Variables: ['portfolio', 'volatility']
Equation hint: var_95
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:12:05,682 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: portfolio, volatility
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: portfolio * (volatility * 1.645)
   R²: 1.0000
2026-01-24 13:12:29,426 - INFO -    Result: portfolio * (volatility * 1.645)
2026-01-24 13:12:29,426 - INFO -    R² = 1.0000
2026-01-24 13:12:29,427 - INFO -    [BEST] New best!
2026-01-24 13:12:29,427 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: portfolio * (volatility * 1.645)
   R² Score: 1.0000
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 95.5/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=1.0000, Extrap: 0.0% ✓

[14/15] liquidation_long
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:12:37,262 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.9999, Extrap: 86.7% ✓
   Running Hybrid System v40... 2026-01-24 13:12:38,306 - INFO - ======================================================================
2026-01-24 13:12:38,306 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:12:38,307 - INFO - ======================================================================
2026-01-24 13:12:38,307 - INFO - Domain: defi_risk
2026-01-24 13:12:38,308 - INFO - Discovery mode: calibrated
2026-01-24 13:12:38,308 - INFO - Primary LLM: anthropic
2026-01-24 13:12:38,309 - INFO - Auto-config: True
2026-01-24 13:12:38,309 - INFO - Max retries: 5
2026-01-24 13:12:38,310 - INFO - Variable sanitization: ENABLED
2026-01-24 13:12:38,310 - INFO - ======================================================================
2026-01-24 13:12:38,311 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:12:39,232 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Liquidation Price LONG: liq = entry*(1 - 1/(L*0.8))
Domain: DEFI_RISK
Samples: 200
Variables: ['entry_price', 'leverage']
Equation hint: liquidation_long
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:12:39,233 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: entry_price, leverage
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: entry_price - (entry_price * (1.25 / leverage))
   R²: 1.0000
2026-01-24 13:13:03,478 - INFO -    Result: entry_price - (entry_price * (1.25 / leverage))
2026-01-24 13:13:03,479 - INFO -    R² = 1.0000
2026-01-24 13:13:03,479 - INFO -    [BEST] New best!
2026-01-24 13:13:03,479 - INFO -    [EARLY STOP] Excellent result

[OK] Discovery complete
   Expression: entry_price - (entry_price * (1.25 / leverage))
   R² Score: 1.0000
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 94.0/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=1.0000, Extrap: 0.0% ✓

[15/15] portfolio_var
   🎯 Generating extrapolation data... ✓
   Running Pure LLM... 2026-01-24 13:13:11,423 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['compiled_function', 'variable_names']

      [DEBUG] Cache keys: ['compiled_function', 'variable_names']
      [DEBUG] Cached function: True
R²=1.0000 ✓
   Running Neural Network...
      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['model', 'scaler_X', 'scaler_y']
R²=0.9989, Extrap: 124.6% ✓
   Running Hybrid System v40... 2026-01-24 13:13:12,476 - INFO - ======================================================================
2026-01-24 13:13:12,476 - INFO - HybridDiscoverySystem v4.2.1 - VARIABLE NAME FIX
2026-01-24 13:13:12,477 - INFO - ======================================================================
2026-01-24 13:13:12,477 - INFO - Domain: defi_risk
2026-01-24 13:13:12,478 - INFO - Discovery mode: calibrated
2026-01-24 13:13:12,478 - INFO - Primary LLM: anthropic
2026-01-24 13:13:12,479 - INFO - Auto-config: True
2026-01-24 13:13:12,479 - INFO - Max retries: 5
2026-01-24 13:13:12,480 - INFO - Variable sanitization: ENABLED
2026-01-24 13:13:12,480 - INFO - ======================================================================
2026-01-24 13:13:12,481 - INFO - Using provided iterations: 50
✅ Anthropic Provider initialized
   Model: claude-sonnet-4-20250514
   Max tokens: 4096
   ✅ Using model: models/gemini-2.5-flash
✅ Google Provider initialized
   Model: models/gemini-2.5-flash
   Max output tokens: 8192
2026-01-24 13:13:13,391 - INFO - [OK] HybridDiscoverySystem v4.2.1 initialized


======================================================================
DISCOVERY WORKFLOW v4.2.1
======================================================================
Description: Portfolio VaR: sqrt(var1² + var2² + 2ρ*var1*var2)
Domain: DEFI_RISK
Samples: 200
Variables: ['var1', 'var2', 'rho']
Equation hint: portfolio_var
======================================================================

[DISCOVER] Running symbolic regression...
2026-01-24 13:13:13,392 - INFO -
[SYMBOLIC] Attempt 1/5 (seed=42)

[DISCOVERY] Starting symbolic regression...
   Variables: var1, var2, rho
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: var1 + (((var2 / ((var2 * rho) + ((var1 / 0.6274719) + var2))) + (rho * 0.76990724)) * var2)
   R²: 0.9949
2026-01-24 13:13:39,274 - INFO -    Result: var1 + (((var2 / ((var2 * rho) + ((var1 / 0.6274719) + var2))) + (rho * 0.76990724)) * var2)
2026-01-24 13:13:39,274 - INFO -    R² = 0.9949
2026-01-24 13:13:39,274 - WARNING -    [WARNING] Possible overfit
2026-01-24 13:13:39,275 - WARNING -       High complexity (92) but R²=0.9949
2026-01-24 13:13:39,275 - INFO -    [BEST] New best!
2026-01-24 13:13:39,275 - INFO -
[SYMBOLIC] Attempt 2/5 (seed=43)

[DISCOVERY] Starting symbolic regression...
   Variables: var1, var2, rho
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: ((var2 * -0.8410067) - var1) * ((rho * -0.28265035) + -0.7988201)
   R²: 0.9709
2026-01-24 13:14:06,489 - INFO -    Result: ((var2 * -0.8410067) - var1) * ((rho * -0.28265035) + -0.7988201)
2026-01-24 13:14:06,489 - INFO -    R² = 0.9709
2026-01-24 13:14:06,490 - WARNING -    [WARNING] Possible overfit
2026-01-24 13:14:06,490 - WARNING -       High complexity (65) but R²=0.9709
2026-01-24 13:14:06,490 - INFO -
[SYMBOLIC] Attempt 3/5 (seed=44)

[DISCOVERY] Starting symbolic regression...
   Variables: var1, var2, rho
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: ((((var2 * -1.8384387) - ((0.7708633 - (-870.9144 / var2)) * var1)) * (rho + 4.1927905)) / -4.5137615) - var2
   R²: 0.9845
2026-01-24 13:14:33,775 - INFO -    Result: ((((var2 * -1.8384387) - ((0.7708633 - (-870.9144 / var2)) * var1)) * (rho + 4.1927905)) / -4.5137615) - var2
2026-01-24 13:14:33,775 - INFO -    R² = 0.9845
2026-01-24 13:14:33,776 - WARNING -    [WARNING] Possible overfit
2026-01-24 13:14:33,777 - WARNING -       High complexity (109) but R²=0.9845
2026-01-24 13:14:33,777 - INFO -
[SYMBOLIC] Attempt 4/5 (seed=45)

[DISCOVERY] Starting symbolic regression...
   Variables: var1, var2, rho
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: (var1 + ((var1 * (var2 * (((18887.658 / var1) + rho) / (var1 + 15131.131)))) + -3646.9062)) / 0.95363194
   R²: 0.9949
2026-01-24 13:15:03,532 - INFO -    Result: (var1 + ((var1 * (var2 * (((18887.658 / var1) + rho) / (var1 + 15131.131)))) + -3646.9062)) / 0.95363194
2026-01-24 13:15:03,533 - INFO -    R² = 0.9949
2026-01-24 13:15:03,533 - WARNING -    [WARNING] Possible overfit
2026-01-24 13:15:03,537 - WARNING -       High complexity (104) but R²=0.9949
2026-01-24 13:15:03,538 - WARNING -       Suspicious constants: ['18887.658', '15131.131', '3646.9062']
2026-01-24 13:15:03,538 - INFO -
[SYMBOLIC] Attempt 5/5 (seed=46)

[DISCOVERY] Starting symbolic regression...
   Variables: var1, var2, rho
   Samples: 200
   Iterations: 50
/home/agagora/Downloads/py312/lib/python3.12/site-packages/pysr/sr.py:1873: UserWarning: Note: Setting `random_state` without also setting `deterministic=True` and `parallelism='serial'` will result in non-deterministic searches.
  warnings.warn(
   ✅ Found: ((var1 * 800.7706) / var2) + (((rho / 3.6658387) + 0.71175313) * (var2 + var1))
   R²: 0.9836
2026-01-24 13:15:44,342 - INFO -    Result: ((var1 * 800.7706) / var2) + (((rho / 3.6658387) + 0.71175313) * (var2 + var1))
2026-01-24 13:15:44,343 - INFO -    R² = 0.9836
2026-01-24 13:15:44,343 - WARNING -    [WARNING] Possible overfit
2026-01-24 13:15:44,344 - WARNING -       High complexity (79) but R²=0.9836
2026-01-24 13:15:44,345 - INFO -
[SUCCESS] SymbolicEngine succeeded (R²=0.9949)

[OK] Discovery complete
   Expression: var1 + (((var2 / ((var2 * rho) + ((var1 / 0.6274719) + var2))) + (rho * 0.76990724)) * var2)
   R² Score: 0.9949
   Engine: symbolic
   Attempt: 1/5

[VALIDATE] Checking expression quality...
[OK] Validation complete
   Score: 70.0/100

======================================================================
[OK] WORKFLOW COMPLETE
======================================================================


      [DEBUG] Testing near extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing medium extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']

      [DEBUG] Testing far extrapolation...
      [DEBUG] Cache keys: ['expression', 'variable_names', 'sanitizer']
R²=0.9949, Extrap: 0.0% ✓

────────────────────────────────────────────────────────────────────────────────
DOMAIN SUMMARY: DEFI_RISK
────────────────────────────────────────────────────────────────────────────────
  Pure LLM                       Success: 3/3 (100.0%), Avg R²: 1.0000
  Neural Network                 Success: 3/3 (100.0%), Avg R²: 0.9995, Avg Extrap:  450.3%
  Hybrid System v40              Success: 3/3 (100.0%), Avg R²: 0.9983, Avg Extrap:    0.0%

================================================================================
                              ALL DOMAINS COMPLETE
================================================================================


================================================================================
                          FINAL SUMMARY - ALL DOMAINS
================================================================================

📊 Total tests: 15

────────────────────────────────────────────────────────────────────────────────
METHOD COMPARISON
────────────────────────────────────────────────────────────────────────────────

🏆 Success Rate:
   Pure LLM                             15/15  (100.0%) ████████████████████
   Neural Network                       15/15  (100.0%) ████████████████████
   Hybrid System v40                    14/15  ( 93.3%) ██████████████████

📈 Average R² (successful tests):
   Pure LLM                            1.0000 ± 0.0000
   Hybrid System v40                   0.9995 ± 0.0013
   Neural Network                      0.9399 ± 0.1789

⏱  Average Time:
   Neural Network                         2.5s
   Pure LLM                               6.9s
   Hybrid System v40                     48.4s

🥇 Wins (best R² per test):
   Pure LLM                             15/15  (100.0%)
   Neural Network                        0/15  (  0.0%)
   Hybrid System v40                     0/15  (  0.0%)

────────────────────────────────────────────────────────────────────────────────
EXTRAPOLATION PERFORMANCE
────────────────────────────────────────────────────────────────────────────────

Near Extrapolation (1.2×):
   Hybrid System v40                      0.0% ±   0.0%  ✅ EXCELLENT  (14/14)
   Neural Network                       175.5% ± 293.1%  ⚠  MODERATE  (13/15)
   Pure LLM                            N/A (0 valid predictions)

Medium Extrapolation (2.0×):
   Hybrid System v40                      0.0% ±   0.0%  ✅ EXCELLENT  (14/14)
   Neural Network                      1231.0% ± 2123.4%  💥 CATASTROPHIC  (13/15)
   Pure LLM                            N/A (0 valid predictions)

Far Extrapolation (5.0×):
   Hybrid System v40                      0.0% ±   0.0%  ✅ EXCELLENT  (14/14)
   Neural Network                       355.0% ± 321.1%  ✗ POOR  (9/15)
   Pure LLM                            N/A (0 valid predictions)

────────────────────────────────────────────────────────────────────────────────
PERFORMANCE BY DOMAIN
────────────────────────────────────────────────────────────────────────────────

CHEMISTRY:
   Hybrid System v40              3/3 (100.0%), R²: 0.9993
   Neural Network                 3/3 (100.0%), R²: 0.9993
   Pure LLM                       3/3 (100.0%), R²: 1.0000

BIOLOGY:
   Hybrid System v40              3/3 (100.0%), R²: 1.0000
   Neural Network                 3/3 (100.0%), R²: 0.9999
   Pure LLM                       3/3 (100.0%), R²: 1.0000

PHYSICS:
   Hybrid System v40              2/3 ( 66.7%), R²: 1.0000
   Neural Network                 3/3 (100.0%), R²: 0.7022
   Pure LLM                       3/3 (100.0%), R²: 1.0000

DEFI_AMM:
   Hybrid System v40              3/3 (100.0%), R²: 1.0000
   Neural Network                 3/3 (100.0%), R²: 0.9985
   Pure LLM                       3/3 (100.0%), R²: 1.0000

DEFI_RISK:
   Hybrid System v40              3/3 (100.0%), R²: 0.9983
   Neural Network                 3/3 (100.0%), R²: 0.9995
   Pure LLM                       3/3 (100.0%), R²: 1.0000

================================================================================

💾 Results saved to: /home/agagora/Downloads/GITHUB/LLM-HypatiaX-Colab/experiments/pyfigures/5_data/results/all_domains_extrap_v4_20260124_131545.json

================================================================================
                          TABLE 1 DATA FOR JMLR PAPER
================================================================================

LaTeX table data:
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Accuracy (R²)} & \textbf{Extrap. Error} & \textbf{Correct Form} & \textbf{Time} \\
\midrule
Hybrid System v40 & $1.00 \pm 0.00$ & 0\% & 14/15 (93.3\%) & 48.4s \\
Neural Network & $0.94 \pm 0.18$ & 1231\% & 15/15 (100.0\%) & 2.5s \\
Pure LLM & $1.00 \pm 0.00$ & 0\% & 15/15 (100.0\%) & 6.9s \\
\bottomrule
\end{tabular}

================================================================================


✅ Complete!


┌──(py312)(agagora㉿localhost)-[~/Downloads/GITHUB/LLM-HypatiaX-Colab/experiments/pyfigures/5_data]
└─$
