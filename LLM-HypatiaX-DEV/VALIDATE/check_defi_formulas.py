import json, re

files = [
    ('hypatiax/data/results/comparison_results/noise-noiseless/noiseless/defi/hypatiax_defi_benchmark_v3_results_seed42.json', 'list_of_tests'),
    ('hypatiax/data/results/comparison_results/noise-noiseless/noiseless/defi_pca/hypatiax_defi_benchmark_pca_results.json', 'list_of_tests'),
    ('hypatiax/data/results/hybrid_pysr/defi/hybrid_defi_20260715_210128.json', 'dict_with_results'),
]

def extract_formulas(fp, shape):
    d = json.load(open(fp))
    out = []
    if shape == 'list_of_tests':
        for rec in d:
            eq = rec.get('equation_id', '')
            results = rec.get('results', {})
            if isinstance(results, dict):
                for method, block in results.items():
                    if isinstance(block, dict):
                        f = block.get('formula') or block.get('python_code') or ''
                        if f:
                            out.append((eq, method, f))
    else:
        results = d.get('results', {})
        if isinstance(results, list):
            for rec in results:
                eq = rec.get('equation_id', rec.get('description', ''))
                f = rec.get('formula') or rec.get('python_code') or ''
                if f:
                    out.append((eq, rec.get('method', ''), f))
        elif isinstance(results, dict):
            for method, recs in results.items():
                if isinstance(recs, list):
                    for rec in recs:
                        eq = rec.get('equation_id', rec.get('description', ''))
                        f = rec.get('formula') or rec.get('python_code') or ''
                        if f:
                            out.append((eq, method, f))
    return out

for fp, shape in files:
    print(f"=== {fp} ===")
    formulas = extract_formulas(fp, shape)
    print(f"  total formula strings found: {len(formulas)}")

    caret_hits = [(eq, m, f) for eq, m, f in formulas if '^' in f]
    bracket_hits = [(eq, m, f) for eq, m, f in formulas if re.search(r'\[X-normalis', f)]
    safe_hits = [(eq, m, f) for eq, m, f in formulas if 'safe_' in f]

    print(f"  formulas containing '^' : {len(caret_hits)}")
    for eq, m, f in caret_hits[:5]:
        print(f"    [{m}] {eq}: {f[:80]}")
    print(f"  formulas containing '[X-normalis...' : {len(bracket_hits)}")
    for eq, m, f in bracket_hits[:5]:
        print(f"    [{m}] {eq}: {f[:80]}")
    print(f"  formulas containing 'safe_' : {len(safe_hits)}")
    for eq, m, f in safe_hits[:5]:
        print(f"    [{m}] {eq}: {f[:80]}")
    print()
