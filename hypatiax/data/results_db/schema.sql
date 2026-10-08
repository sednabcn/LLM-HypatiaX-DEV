PRAGMA foreign_keys=ON;

-- ===== Pipeline layer (from run_all.sh) =====
CREATE TABLE experiment(
  experiment_id TEXT PRIMARY KEY,          -- run_all.sh step name
  step_order    INTEGER,                   -- position in _STEP_ORDER
  description   TEXT,
  paper_refs    TEXT,                      -- Tab/Fig/§ mentioned in the step description
  produces_results INTEGER NOT NULL,       -- 1 = writes files under results/
  notes         TEXT);
CREATE TABLE script(script_id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
CREATE TABLE experiment_script(
  experiment_id TEXT REFERENCES experiment, script_id INTEGER REFERENCES script,
  PRIMARY KEY(experiment_id,script_id));
CREATE TABLE experiment_param(               -- CLI flags / env defaults found in run_all.sh
  experiment_id TEXT, param TEXT, value TEXT, PRIMARY KEY(experiment_id,param,value));

-- ===== File inventory layer (from tree_results.txt; metadata parsed from path/filename) =====
CREATE TABLE result_file(
  file_id INTEGER PRIMARY KEY,
  rel_path TEXT UNIQUE NOT NULL, dir_path TEXT, file_name TEXT, ext TEXT,
  experiment_id TEXT REFERENCES experiment,
  mapping_basis TEXT,                      -- how the file was tied to an experiment
  file_kind TEXT,                          -- checkpoint, protocol_run, benchmark_results, rescued_shard, ...
  shard INTEGER, n_shards INTEGER, seed INTEGER, temperature REAL, run_no INTEGER,
  run_timestamp TEXT,                      -- parsed from YYYYMMDD_HHMMSS in the name
  is_rescued INTEGER DEFAULT 0, rescued_epoch INTEGER, rescued_base_name TEXT,
  is_partial INTEGER DEFAULT 0, is_saved_copy INTEGER DEFAULT 0,
  feynman_domain TEXT, benchmark_version TEXT, split_protocol TEXT, phase TEXT,
  content_loaded INTEGER DEFAULT 0);        -- set to 1 by ingest_json.py
CREATE INDEX ix_rf_exp ON result_file(experiment_id);
CREATE INDEX ix_rf_kind ON result_file(file_kind);
CREATE INDEX ix_rf_seed ON result_file(seed);

-- ===== Content layer (filled by ingest_json.py from the real JSON files) =====
CREATE TABLE json_doc(                       -- one row per ingested JSON
  file_id INTEGER PRIMARY KEY REFERENCES result_file, sha256 TEXT, size_bytes INTEGER,
  top_level_type TEXT, n_top_keys INTEGER, parse_error TEXT);
CREATE TABLE json_item(                      -- every leaf value: schema-agnostic EAV
  item_id INTEGER PRIMARY KEY, file_id INTEGER REFERENCES result_file,
  json_path TEXT NOT NULL,                   -- e.g. $.results[3].r2
  parent_path TEXT, key TEXT, depth INTEGER,
  vtype TEXT, v_num REAL, v_text TEXT, v_bool INTEGER);
CREATE INDEX ix_ji_file ON json_item(file_id);
CREATE INDEX ix_ji_key  ON json_item(key);
CREATE TABLE metric(                         -- "relevant items": leaves whose key looks like a result metric
  metric_id INTEGER PRIMARY KEY, file_id INTEGER REFERENCES result_file,
  record_path TEXT,                          -- path of the enclosing record (task/equation/case)
  record_label TEXT,                         -- name/id/equation/case found in that record
  method TEXT, metric_name TEXT, value REAL, value_text TEXT);
CREATE INDEX ix_m_name ON metric(metric_name);

-- ===== Views =====
CREATE VIEW v_experiment_summary AS
SELECT e.experiment_id, e.step_order, e.description, e.paper_refs,
       COUNT(f.file_id) n_files,
       SUM(f.ext='json') n_json,
       SUM(f.file_kind IN('benchmark_results','protocol_run','nguyen12_result','hybrid_result','seed_sweep','summary')) n_result_files,
       SUM(f.is_rescued) n_rescued, SUM(f.is_partial) n_partial,
       COUNT(DISTINCT f.seed) n_seeds, MIN(f.run_timestamp) first_ts, MAX(f.run_timestamp) last_ts,
       SUM(f.content_loaded) n_loaded
FROM experiment e LEFT JOIN result_file f USING(experiment_id)
GROUP BY e.experiment_id ORDER BY e.step_order;
CREATE VIEW v_seed_coverage AS
SELECT experiment_id, seed, COUNT(*) n_files, GROUP_CONCAT(DISTINCT file_kind) kinds
FROM result_file WHERE seed IS NOT NULL GROUP BY experiment_id, seed;
CREATE VIEW v_rescued_vs_original AS      -- each rescued shard next to its base file
SELECT r.file_id rescued_id, r.rel_path rescued_path, r.shard, r.rescued_epoch,
       datetime(r.rescued_epoch,'unixepoch') rescued_at, o.file_id original_id, o.rel_path original_path
FROM result_file r LEFT JOIN result_file o
  ON o.dir_path=r.dir_path AND o.is_rescued=0 AND
     (o.file_name=r.rescued_base_name || '.json' OR o.file_name=r.rescued_base_name)
WHERE r.is_rescued=1;
CREATE VIEW v_unmapped_files AS SELECT * FROM result_file WHERE experiment_id IS NULL;
