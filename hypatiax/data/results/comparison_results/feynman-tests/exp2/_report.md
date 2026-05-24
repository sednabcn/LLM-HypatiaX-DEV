
# HypatiaX Analysis Report — `exp2_feynman`

Experiment mode: **multi_method**
N total: 210 | N standard: 210 | N intractable: 0
R² success threshold: 0.8

> **Multi-method experiment**: up to six method keys may be present in the raw output (`PureLLM Baseline`, `ImprovedNN`, `EnhancedHybridSystemDeFi`, `HybridSystemLLMNN all-domains`, `SymbolicEngineWithLLM`, `HybridDiscoverySystem v50_2`). `merge_shards.py` normalises these to canonical slugs; only `pure_llm`, `neural_network`, and `hybrid` are included in METHODS and drive all statistical comparisons. The remaining keys (`hybrid_all_domains`, `symbolic_engine`, `hybrid_v50_2`) are present in records but excluded from MW tests and method summary tables. For `exp2_feynman`: `symbolic_engine` (SymbolicEngineWithLLM) and any PySR-only variants are retained in records for completeness but not compared.

## ✅ No Fatal Conditions


## ℹ️ Informational / Warnings

- WARN_MULTI_METHOD: up to six method keys may be present in records (pure_llm, neural_network, hybrid, hybrid_all_domains, symbolic_engine, hybrid_v50_2). Only METHODS = [pure_llm, neural_network, hybrid] drive MW tests and summary tables. Extra keys (symbolic_engine / hybrid_all_domains / hybrid_v50_2) are retained in records for completeness but excluded from all statistical comparisons.

## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 30 | 100.0% | 100.0% | 1.0000 | 1.0000 |
| Neural Net | 30 | 100.0% | 100.0% | 0.9999 | 0.9993 |
| Hybrid | 30 | 100.0% | 100.0% | 1.0000 | 0.9999 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=23.0,  p=0.0000**,  direction=b_greater,  n=(30, 30)

### Hybrid vs Neural Net

  U=839.0,  p=0.0000**,  direction=a_greater,  n=(30, 30)

### Neural Net vs Pure LLM

  U=0.0,  p=0.0000**,  direction=b_greater,  n=(30, 30)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 30
Hybrid wins:  28  (93.3%)
NN wins:      2
Tied:         0

## Coverage Gaps (180 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| benchmark_results_30 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_31 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_32 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_33 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_34 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_35 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_36 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_37 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_38 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_39 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_40 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_41 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_42 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_43 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_44 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_45 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_46 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_47 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_48 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_49 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_50 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_51 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_52 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_53 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_54 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_55 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_56 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_57 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_58 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_59 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_60 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_61 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_62 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_63 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_64 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_65 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_66 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_67 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_68 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_69 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_70 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_71 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_72 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_73 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_74 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_75 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_76 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_77 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_78 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_79 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_80 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_81 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_82 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_83 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_84 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_85 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_86 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_87 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_88 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_89 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_90 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_91 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_92 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_93 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_94 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_95 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_96 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_97 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_98 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_99 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_100 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_101 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_102 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_103 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_104 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_105 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_106 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_107 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_108 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_109 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_110 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_111 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_112 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_113 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_114 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_115 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_116 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_117 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_118 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_119 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_120 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_121 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_122 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_123 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_124 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_125 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_126 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_127 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_128 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_129 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_130 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_131 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_132 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_133 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_134 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_135 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_136 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_137 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_138 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_139 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_140 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_141 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_142 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_143 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_144 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_145 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_146 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_147 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_148 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_149 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_150 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_151 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_152 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_153 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_154 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_155 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_156 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_157 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_158 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_159 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_160 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_161 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_162 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_163 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_164 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_165 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_166 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_167 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_168 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_169 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_170 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_171 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_172 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_173 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_174 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_175 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_176 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_177 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_178 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_179 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_180 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_181 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_182 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_183 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_184 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_185 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_186 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_187 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_188 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_189 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_190 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_191 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_192 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_193 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_194 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_195 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_196 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_197 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_198 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_199 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_200 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_201 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_202 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_203 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_204 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_205 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_206 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_207 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_208 | None | None | N/A | N/A | N/A | N/A |
| benchmark_results_209 | None | None | N/A | N/A | N/A | N/A |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| unknown | 30 | 100.0% | 100.0% | 100.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| unknown | 30 | 1.0000 | 0.9999 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | N/A | N/A | 0 |
| Neural Net | N/A | N/A | 0 |
| Hybrid | N/A | N/A | 0 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 5.6873 | 7.2148 | 170.62 | 30 |
| Neural Net | 1.0871 | 1.1198 | 32.61 | 30 |
| Hybrid | 5.6688 | 6.5898 | 170.06 | 30 |

## Hybrid Routing Decisions

_No hybrid decision data available._
