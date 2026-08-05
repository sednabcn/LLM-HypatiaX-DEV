# BLOG_LINK_GUIDE.md — Connecting the GitHub Repo to the Blog

> How to link **`LLM-HypatiaX-PAPERS-REPRODUCIBILITY`** ↔ **`https://sednabcn.github.io/ai-llm-blog`**

---

## Overview

There are two complementary directions to link:

- **Repo → Blog**: add badges and links in `README.md` that point readers to the blog
- **Blog → Repo**: add a post (or update an existing one) that links back to this GitHub repo

Both are described below.

---

## 1. Repo → Blog (already in README.md)

The `README.md` in this package already contains:

```markdown
[![Blog](https://img.shields.io/badge/Blog-sednabcn.github.io-green)](https://sednabcn.github.io/ai-llm-blog)
```

and a **Blog & Extended Discussion** section:

```markdown
## Blog & Extended Discussion

Extended methodology, intuitions, and result commentary are published at:

**[https://sednabcn.github.io/ai-llm-blog](https://sednabcn.github.io/ai-llm-blog)**

The blog post corresponding to this paper is:
`https://sednabcn.github.io/ai-llm-blog/posts/hypatiax-symbolic-regression/`
```

**Action required**: once you publish the blog post, update the URL above to match the exact slug you choose.

---

## 2. Blog → Repo: Writing the Linked Post

The blog at `https://sednabcn.github.io/ai-llm-blog` is almost certainly a **Jekyll** or **Hugo** static site served from a `gh-pages` branch or a `docs/` folder of a separate repo (`sednabcn/ai-llm-blog` or similar).

### Step-by-step

#### a) Find your blog's posts directory

```bash
# If the blog is a separate repo:
git clone https://github.com/sednabcn/ai-llm-blog
cd ai-llm-blog
ls _posts/       # Jekyll
# or
ls content/posts/  # Hugo
```

#### b) Create a new post file

For **Jekyll** (filename format: `YYYY-MM-DD-slug.md`):

```bash
touch _posts/2026-05-12-hypatiax-symbolic-regression.md
```

For **Hugo**:

```bash
hugo new posts/hypatiax-symbolic-regression.md
```

#### c) Post front matter and content template

Paste this into the new file and fill in the blanks:

```markdown
---
layout: post          # Jekyll; remove this line for Hugo
title: "HypatiaX: LLM-Guided Hybrid Symbolic Regression — Paper & Code"
date: 2026-05-12
categories: [research, symbolic-regression, llm]
tags: [hypatiax, jmlr, reproducibility, pysr, defi]
description: >
  We present HypatiaX, a hybrid LLM + symbolic regression system evaluated
  across 31 physics/science equations and 25 DeFi formulas.
  Full reproducibility package available on GitHub.
---

## Abstract

_Paste or paraphrase the paper abstract here._

## Key Results

- Exp1 (Ablation): ...
- Exp2 (31-equation benchmark): ...
- DeFi domain: ...

## Reproducibility

All code, pre-computed results, and step-by-step guides are available in the
GitHub repository:

> **[github.com/<your-org>/LLM-HypatiaX-PAPERS-REPRODUCIBILITY](https://github.com/<your-org>/LLM-HypatiaX-PAPERS-REPRODUCIBILITY)**

To reproduce results:

```bash
git clone https://github.com/<your-org>/LLM-HypatiaX-PAPERS-REPRODUCIBILITY
cd LLM-HypatiaX-PAPERS-REPRODUCIBILITY
bash setup_environment.sh && source activate_hypatiax.sh
bash run_all.sh
```

Pre-computed results are in `hypatiax/data/results_v1/`.

## Citation

```bibtex
@article{hypatiax2026,
  title   = {LLM-Guided Hybrid Symbolic Regression},
  author  = {<Authors>},
  journal = {JMLR},
  year    = {2026}
}
```
```

#### d) Commit and push

```bash
git add _posts/2026-05-12-hypatiax-symbolic-regression.md
git commit -m "Add HypatiaX paper post with GitHub repo link"
git push origin main   # or gh-pages / master depending on your setup
```

GitHub Pages will rebuild automatically within ~1 minute.

---

## 3. Mutual Cross-Reference Table

Once both sides are live, confirm these URLs resolve correctly:

| From | To | URL |
|---|---|---|
| GitHub README badge | Blog homepage | `https://sednabcn.github.io/ai-llm-blog` |
| GitHub README section | Blog post | `https://sednabcn.github.io/ai-llm-blog/posts/hypatiax-symbolic-regression/` |
| Blog post body | GitHub repo | `https://github.com/<org>/LLM-HypatiaX-PAPERS-REPRODUCIBILITY` |
| Blog post body | Reproducibility guide HTML | `https://github.com/<org>/LLM-HypatiaX-PAPERS-REPRODUCIBILITY/blob/main/hypatiax/reproducibility/HypatiaX_Step_By_Step_Guide.html` |

---

## 4. Optional: GitHub Social Preview Card

Add a social preview image so the repo shows nicely when shared or linked from the blog:

1. Go to your GitHub repo → **Settings** → **Social preview**
2. Upload a 1280×640 PNG (e.g., a banner with "HypatiaX", equation examples, JMLR logo)

This makes the blog's link card (Open Graph preview) display a rich image automatically.

---

## 5. Optional: GitHub Topics / About Section

In the repo **About** (gear icon top-right of the repo page):

- **Description**: `Reproducibility package for "LLM-Guided Hybrid Symbolic Regression" (JMLR 2026)`
- **Website**: `https://sednabcn.github.io/ai-llm-blog`
- **Topics**: `symbolic-regression`, `llm`, `reproducibility`, `jmlr`, `pysr`, `defi`, `scientific-machine-learning`

This ensures the blog URL appears prominently on the GitHub repo page.

---

## 6. JMLR Supplementary Material Note

JMLR allows listing a supplementary URL. In your submission system, list:

```
Code & Data: https://github.com/<org>/LLM-HypatiaX-PAPERS-REPRODUCIBILITY
Blog:        https://sednabcn.github.io/ai-llm-blog
```

Reviewers can then navigate from the paper → repo → blog seamlessly.

---

*Last updated: May 2026*
