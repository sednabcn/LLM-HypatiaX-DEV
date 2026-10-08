#!/usr/bin/env python3
"""Load the CONTENTS of the real result JSONs into hypatiax_experiments.db.
Run from anywhere:  python3 ingest_json.py hypatiax_experiments.db /path/to/hypatiax/data/results
Matches files by relative path to result_file.rel_path, then fills json_doc, json_item (every leaf),
and metric (leaves whose key looks like a result metric). Re-runnable (reloads a file's rows)."""
import sys, os, json, re, hashlib, sqlite3
db_path, root = sys.argv[1], sys.argv[2]
db = sqlite3.connect(db_path); c = db.cursor()
METRIC = re.compile(r'(^|_)(r2|r_2|rmse|mae|mse|nrmse|error|err|rel_err|success|solved|accuracy|score|rank|time|runtime|'
                    r'elapsed|p_value|pvalue|statistic|u_stat|ci_low|ci_high|mean|std|median|n_|count|complexity|loss|instability)($|_)', re.I)
LABEL_KEYS = ('name','equation','equation_name','case','case_id','task','task_name','id','problem','benchmark','domain','formula')
METHOD_KEYS = ('method','system','model','approach','algorithm')

def leaves(o, path='$', parent=None, key=None, depth=0):
    if isinstance(o, dict):
        for k,v in o.items(): yield from leaves(v, f'{path}.{k}', path, k, depth+1)
    elif isinstance(o, list):
        for i,v in enumerate(o): yield from leaves(v, f'{path}[{i}]', path, key, depth+1)
    else: yield path, parent, key, depth, o

def records(o, path='$'):
    """yield (path, dict) for every dict that contains at least one scalar"""
    if isinstance(o, dict):
        if any(not isinstance(v,(dict,list)) for v in o.values()): yield path, o
        for k,v in o.items(): yield from records(v, f'{path}.{k}')
    elif isinstance(o, list):
        for i,v in enumerate(o): yield from records(v, f'{path}[{i}]')

ok=bad=missing=0
for fid, rel in c.execute("SELECT file_id, rel_path FROM result_file WHERE ext='json'").fetchall():
    fp = os.path.join(root, rel)
    if not os.path.exists(fp): missing+=1; continue
    for t in ('json_item','metric','json_doc'): c.execute(f'DELETE FROM {t} WHERE file_id=?',(fid,))
    raw = open(fp,'rb').read()
    sha = hashlib.sha256(raw).hexdigest()
    try: data = json.loads(raw)
    except Exception as e:
        c.execute('INSERT INTO json_doc VALUES (?,?,?,?,?,?)',(fid,sha,len(raw),None,None,str(e)[:200])); bad+=1; continue
    c.execute('INSERT INTO json_doc VALUES (?,?,?,?,?,?)',(fid,sha,len(raw),type(data).__name__,
              len(data) if isinstance(data,(dict,list)) else None,None))
    rows=[]
    for p,par,k,d,v in leaves(data):
        vt = 'null' if v is None else 'bool' if isinstance(v,bool) else 'num' if isinstance(v,(int,float)) else 'text'
        rows.append((fid,p,par,k,d,vt, float(v) if vt=='num' else None, str(v)[:2000] if vt=='text' else None,
                     int(v) if vt=='bool' else None))
    c.executemany('INSERT INTO json_item(file_id,json_path,parent_path,key,depth,vtype,v_num,v_text,v_bool) VALUES (?,?,?,?,?,?,?,?,?)',rows)
    mrows=[]
    for rp,rec in records(data):
        label = next((str(rec[k]) for k in LABEL_KEYS if k in rec and not isinstance(rec[k],(dict,list))),None)
        meth  = next((str(rec[k]) for k in METHOD_KEYS if k in rec and not isinstance(rec[k],(dict,list))),None)
        for k,v in rec.items():
            if isinstance(v,(dict,list)) or v is None: continue
            if isinstance(v,(int,float)) and not isinstance(v,bool) and METRIC.search(k):
                mrows.append((fid,rp,label,meth,k,float(v),None))
    c.executemany('INSERT INTO metric(file_id,record_path,record_label,method,metric_name,value,value_text) VALUES (?,?,?,?,?,?,?)',mrows)
    c.execute('UPDATE result_file SET content_loaded=1 WHERE file_id=?',(fid,)); ok+=1
db.commit()
print(f'loaded={ok} parse_errors={bad} not_found_under_root={missing}')
