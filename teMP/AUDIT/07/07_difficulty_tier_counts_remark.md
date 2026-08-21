# Issue 7 — Remark on Difficulty-Tier Counts: 24/27/20 → 24/29/21 vs. "+3 Hard"

**Category:** [TEXT — pure writing fix] — pure text/arithmetic fix

**Location:** `jmlr_paper_main.tex`, `subsec:difficulty-classification`, unlabeled remark following the Easy/Medium/Hard definitions, l. 656–660.

## What is wrong
The tier definitions immediately above state Easy n = 24, Medium n = 29, Hard n = 21 (totalling 74). The remark directly below reads:

> An earlier version of this benchmark comprised 71 tasks (Easy=24, Medium=27, Hard=20). The current benchmark (v3.0, 74 tasks) adds three additional hard cases in multi-asset risk: correlated portfolio VaR, component ES, and multi-collateral LTV.

71+3 = 74 is correct in total, but 24/27/20 → 24/29/21 is a change of +0/+2/+1, not +0/+0/+3. The remark's claim that all three additions are Hard-tier is arithmetically inconsistent with the tier counts stated one paragraph above it.

## Fix required
Either the remark's "adds three additional hard cases" is wrong (should read something like "+2 Medium, +1 Hard"), or the Medium/Hard counts in `subsec:difficulty-classification` are wrong. Purely a text correction; no code or data investigation implicated.
