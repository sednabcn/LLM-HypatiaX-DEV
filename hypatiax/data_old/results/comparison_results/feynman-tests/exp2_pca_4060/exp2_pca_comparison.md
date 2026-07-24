# exp2 Feynman Benchmark — Random 80/20 vs PCA 40/60 Comparison

> **FIX-C3**: The original Feynman benchmark (`exp2`) used a random 80/20 train/test split (`train_test_split(test_size=0.2, random_state=42)`).  
> All DeFi benchmarks use a PCA-directed 40/60 extrapolation split (`build_extrap_split`, `extrap_train_frac=0.6`).  
> These protocols are **not directly comparable**. This table shows both results side-by-side.

## Summary

| Split | Threshold | Solved | Solve Rate |
|-------|-----------|--------|------------|
| Random 80/20 (`random_state=42`) | R²≥0.999999 | 74/90 | 0.822 |
| PCA 40/60 (FIX-C3 corrected)     | R²≥0.999999 | 71/90 | 0.789 |

*Per-equation breakdown not available — run with full result JSON files to populate this section.*


---
Threshold: R² ≥ 0.999999  |  Legacy source: `fixc3_baseline.json`  |  PCA source: `exp2_pca_4060_summary.json`
