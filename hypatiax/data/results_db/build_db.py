#!/usr/bin/env python3
"""Build hypatiax_experiments.db (SQLite) from tree_results.txt + run_all.sh.
Usage: python3 build_db.py [tree_results.txt] [run_all.sh] [out.db]"""
import re, sys, sqlite3, os, datetime
tree = sys.argv[1] if len(sys.argv)>1 else 'tree_results.txt'
sh   = sys.argv[2] if len(sys.argv)>2 else 'run_all.sh'
out  = sys.argv[3] if len(sys.argv)>3 else 'hypatiax_experiments.db'

# ---------- 1. parse tree into relative paths ----------
paths, stack = [], []
for line in open(tree, encoding='utf-8'):
    line = line.replace('\u00a0',' ')
    m = re.match(r'^((?:│   |    )*)(?:├── |└── )(.+?)\s*$', line.rstrip('\n'))
    if not m: continue
    depth = len(m.group(1))//4; name = m.group(2)
    stack = stack[:depth]
    nxt_is_dir = False
    stack.append(name)
    paths.append('/'.join(stack))
# directories = any path that is a prefix of another
dirs = {p for p in paths for q in paths if q.startswith(p+'/')}
files = [p for p in paths if p not in dirs and p != 'tree_results.txt']

# ---------- 2. parse run_all.sh steps ----------
src = open(sh, encoding='utf-8').read().split('\n')
steps = []
for i,l in enumerate(src):
    m = re.match(r'run (\w+) "([^"]*)"', l)
    if m: steps.append((i, m.group(1), m.group(2)))
    elif l.startswith('run audit_nb06'):
        steps.append((i, l.split()[1], src[i+1].strip().strip('"\\ ').strip('"')))
order = re.search(r'_STEP_ORDER="([^"]+)"', '\n'.join(src)).group(1).split()
blocks = {}
for k,(i,n,d) in enumerate(steps):
    end = steps[k+1][0] if k+1<len(steps) else len(src)
    blocks[n] = '\n'.join(src[i:end])
def refs(desc):
    return '; '.join(re.findall(r'(?:Tab|Fig|§|SS)[ .]?[\w\-,\. ]*?\d[\w\.]*', desc)) or None
SCRIPT_RE = re.compile(r'([A-Za-z0-9_]+\.py)')
cfg = {  # step -> (seed(s), noise/protocol note, produces_results)
}
produces = {'exp1','exp1b','portfolio','exp1_ablation','exp1_five','exp1_pca','exp1b_pca','extrap','hybrid_all_domains',
            'instability','exp2_feynman','exp2_feynman_pca_4060','exp2_feynman_extrap','exp2','exp2_five','exp3','exp3b','suppA','suppB','suppB_sc'}

db_exists = os.path.exists(out)
if db_exists: os.remove(out)
db = sqlite3.connect(out); c = db.cursor()
c.executescript(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'schema.sql')).read())

# experiments
for n in order:
    d = next((d for _,s,d in steps if s==n), '')
    c.execute('INSERT INTO experiment VALUES (?,?,?,?,?,?)',
        (n, order.index(n)+1, d, refs(d), int(n in produces), None))
# fix-up experiments not in run_all.sh (found only in results tree)
extra = [('fixn3_item_ii','Nguyen-12 temperature replication, seed 123, temp 0 vs 0.25, runs 1-5 (NOT in run_all.sh)'),
         ('issue2b_experiment','Issue-2b freeze-cache experiment, phaseA runs 1-3 + phaseB full (NOT in run_all.sh)')]
for n,d in extra:
    c.execute('INSERT INTO experiment VALUES (?,?,?,?,?,?)',(n,None,d,None,1,'results-tree only; no run_all.sh step'))

# scripts
for n,b in blocks.items():
    for s in sorted(set(SCRIPT_RE.findall(b))):
        c.execute('INSERT OR IGNORE INTO script(name) VALUES (?)',(s,))
        sid = c.execute('SELECT script_id FROM script WHERE name=?',(s,)).fetchone()[0]
        c.execute('INSERT OR IGNORE INTO experiment_script VALUES (?,?)',(n,sid))

# experiment_param: pull explicit CLI flags / env from each block
for n,b in blocks.items():
    for k,v in re.findall(r'--(seed|temperature|noise|n-shards|n_shards|shards|split|domains?)[ =]([\w\.,\$\{\}"]+)', b):
        c.execute('INSERT OR IGNORE INTO experiment_param VALUES (?,?,?)',(n,'--'+k,v.strip('"')))
# global env from header
for k,v in re.findall(r'^export (\w+)="?\$?\{?[A-Z_]*:?-?([^"}\n]*)"?\}?$', '\n'.join(src[:60]), re.M):
    c.execute('INSERT OR IGNORE INTO experiment_param VALUES (?,?,?)',('_GLOBAL',k,v))

# ---------- 3. directory -> experiment map (prefix match, longest wins) ----------
dmap = [
 ('ablation/exp1_ablation','exp1_ablation','run_all step output dir'),
 ('comparison_results/extrapolation','extrap','run_all step output dir'),
 ('comparison_results/feynman-tests/exp2_extrap','exp2_feynman_extrap','run_all step output dir'),
 ('comparison_results/feynman-tests/exp2_multi','exp2','run_all step output dir'),
 ('comparison_results/feynman-tests/exp2_pca_4060','exp2_feynman_pca_4060','run_all step output dir'),
 ('comparison_results/feynman-tests/exp2','exp2_feynman','run_all step output dir'),
 ('comparison_results/feynman-tests/noise-sweep','suppB','run_all step output dir'),
 ('comparison_results/feynman-tests/sample-complexity','suppB_sc','run_all step output dir'),
 ('comparison_results/noise-noiseless/15_pca','exp1b_pca','run_all step output dir'),
 ('comparison_results/noise-noiseless/15','exp1b','run_all step output dir'),
 ('comparison_results/noise-noiseless/noiseless/defi_pca/multi-seeds','exp1b_pca','inferred from "multi-seeds" + pca'),
 ('comparison_results/noise-noiseless/noiseless/defi_pca','exp1_pca','run_all step output dir'),
 ('comparison_results/noise-noiseless/noiseless/defi/multi-seeds','exp1b','inferred from "multi-seeds"'),
 ('comparison_results/noise-noiseless/noiseless/defi','exp1','run_all step output dir'),
 ('comparison_results/sample_complexity_n0050_shared_checkpoint.json','suppB_sc','inferred from filename'),
 ('comparison_results/benchmark_results.json','exp2_feynman_extrap','inferred (merge_extrap_into_benchmark)'),
 ('extrapolation/multi_seed','exp3b','run_all step output dir'),
 ('extrapolation','exp3','run_all step output dir'),
 ('extrapolation_sparse_seed_off','exp3','inferred variant of extrapolation/'),
 ('extrapolation-sparse_seeds_fallback','exp3','inferred variant of extrapolation/'),
 ('figures','instability','run_all step output dir'),
 ('five_systems/exp1_five','exp1_five','run_all step output dir'),
 ('five_systems/exp2_five','exp2_five','run_all step output dir'),
 ('fixc3_baseline.json','exp2_feynman_pca_4060','run_all: BASELINE path in FIX-C3 step'),
 ('fixn3_item_ii','fixn3_item_ii','results-tree only'),
 ('hybrid_llm_nn','hybrid_all_domains','run_all step output dir'),
 ('hybrid_pysr','suppA','run_all step output dir'),
 ('issue2b_experiment','issue2b_experiment','results-tree only'),
 ('portfolio_variance_audit','portfolio','run_all step output dir'),
]
dmap.sort(key=lambda x:-len(x[0]))
def exp_for(p):
    for pre,e,why in dmap:
        if p==pre or p.startswith(pre+'/'): return e,why
    return None,None

# ---------- 4. file classification ----------
def classify(p):
    f = p.split('/')[-1]; low=f.lower()
    r = dict(kind='other', shard=None, n_shards=None, seed=None, temp=None, run_no=None, ts=None,
             rescued=0, rescued_epoch=None, partial=0, saved_copy=int('/_saved/' in p), domain=None,
             version=None, split=None, phase=None, rescued_base=None)
    m = re.search(r'__shard(\d+)_rescued_(\d+)', f)
    if m:
        r['rescued']=1; r['shard']=int(m.group(1)); r['rescued_epoch']=int(m.group(2))
        r['rescued_base']=f[:m.start()]
        r['kind']='rescued_shard'
    elif f.startswith('_checkpoint_shard') or 'checkpoint_shard' in f: r['kind']='checkpoint'
    elif f.startswith('pca4060_checkpoint_'): r['kind']='checkpoint'
    elif 'checkpoint' in f: r['kind']='checkpoint'
    elif f.startswith('protocol_core_'): r['kind']='protocol_run'
    elif f.startswith('noise_sweep_'): r['kind']='noise_sweep'
    elif f.startswith('sample_complexity_'): r['kind']='sample_complexity'
    elif f.startswith('_exp3') and 'partial' in f: r['kind']='partial'; r['partial']=1
    elif f.startswith('exp3_nguyen12_'): r['kind']='nguyen12_result'
    elif 'benchmark_results' in f or 'benchmark_v' in f or 'benchmark_pca' in f: r['kind']='benchmark_results'
    elif 'summary' in f: r['kind']='summary'
    elif 'provenance' in f: r['kind']='provenance'
    elif 'fixc3_baseline' in f: r['kind']='fixc3_baseline'
    elif 'paper_numbers' in f: r['kind']='paper_numbers'
    elif 'contradiction_audit' in f: r['kind']='audit_report'
    elif 'split_protocol' in f: r['kind']='split_disclosure'
    elif 'mannwhitney' in f or 'instability' in f: r['kind']='statistics'
    elif 'seed_sweep' in f: r['kind']='seed_sweep'
    elif 'wall_clock' in f: r['kind']='diagnostics'
    elif f.endswith('.log'): r['kind']='log'
    elif 'llm_freeze_cache' in f: r['kind']='cache'
    elif 'extrapolation_73cases' in f or f.startswith('hybrid_'): r['kind']='hybrid_result'
    if m is None:
        m2 = re.search(r'shard(\d+)', f)
        if m2 and r['kind'] in ('checkpoint','benchmark_results','audit_report','fixc3_baseline'): r['shard']=int(m2.group(1))
    m = re.search(r'seed(\d+)', f);              r['seed']=int(m.group(1)) if m else None
    m = re.search(r'nshards(\d+)', f);           r['n_shards']=int(m.group(1)) if m else None
    if r['kind'] in ('noise_sweep','sample_complexity','nguyen12_result') and r['n_shards'] and r['shard'] is None: pass
    m = re.search(r'_temp(0p25|0)_?', f)
    if m: r['temp']=0.25 if m.group(1)=='0p25' else 0.0
    m = re.search(r'_run(\d+)', f)
    if m and 'protocol_core' not in f: r['run_no']=int(m.group(1))
    m = re.search(r'(20\d{6})_(\d{6})', f)
    if m: r['ts']=datetime.datetime.strptime(m.group(1)+m.group(2),'%Y%m%d%H%M%S').isoformat(sep=' ')
    m = re.search(r'checkpoint_(feynman_\w+)\.json', f)
    if m: r['domain']=m.group(1)
    m = re.search(r'v3c3|v3c2|_v4_|_v3_|v3c\b', f)
    if m: r['version']=m.group(0).strip('_')
    if '_pca' in f or 'pca_' in f or '/defi_pca' in p or '15_pca' in p or 'pca_4060' in p: r['split']='PCA 40/60'
    m = re.search(r'(phaseA_run\d|phaseB_full)', p)
    if m: r['phase']=m.group(1)
    if 'noisy' in f: r['phase']='noisy'
    elif 'noiseless' in f and 'protocol_core' in f: r['phase']='noiseless'
    return r

file_rows=[]
for p in sorted(files):
    e,why = exp_for(p); r=classify(p)
    ext = os.path.splitext(p)[1].lstrip('.').lower()
    c.execute('''INSERT INTO result_file(rel_path,dir_path,file_name,ext,experiment_id,mapping_basis,file_kind,
        shard,n_shards,seed,temperature,run_no,run_timestamp,is_rescued,rescued_epoch,is_partial,is_saved_copy,
        feynman_domain,benchmark_version,split_protocol,phase,rescued_base_name)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (p,os.path.dirname(p),os.path.basename(p),ext,e,why,r['kind'],r['shard'],r['n_shards'],r['seed'],r['temp'],
         r['run_no'],r['ts'],r['rescued'],r['rescued_epoch'],r['partial'],r['saved_copy'],r['domain'],r['version'],
         r['split'],r['phase'],r['rescued_base']))
db.commit()
print('files:',len(files),'dirs:',len(dirs),'json:',sum(f.endswith('.json') for f in files))
db.close()
