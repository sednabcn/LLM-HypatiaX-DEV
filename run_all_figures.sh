RES=<results_dir>      # where the committed JSON/CSV live
OUT=figures

for exp in exp1_ablation exp1b exp1b_pca exp2_feynman_extrap exp2_feynman_pca \
           exp3 exp3b suppB suppB_sc suppA instability; do
  python3 generate_figures.py --experiment $exp --results-dir $RES --figures-dir $OUT
done
