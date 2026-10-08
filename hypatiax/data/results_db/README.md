# HypatiaX experiments database

`hypatiax_experiments.db` (SQLite). Rebuild: `python3 build_db.py tree_results.txt run_all.sh out.db`.
Load JSON contents (needs the real files): `python3 ingest_json.py hypatiax_experiments.db <results_dir>`.

Tables: experiment, script, experiment_script, experiment_param, result_file (432 files, 410 JSON),
json_doc, json_item (EAV of every leaf), metric (r2/rmse/success/time/... per record). Views: v_experiment_summary,
v_seed_coverage, v_rescued_vs_original, v_unmapped_files.

STATUS: result_file/experiment layers are populated. json_doc/json_item/metric are EMPTY until ingest_json.py is run,
because the uploaded files were directory listings, not the JSON files.
