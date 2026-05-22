# Pool-coverage statistical analysis — Phase 2 §12.6

Bootstrapped recall-vs-pool-size on the §2.17 expanded qrels (4758 positives across 313 cells). B = 200 per fraction. Seed = 0.

**Reading the table**: each row is one variant at one pool fraction. *Mean ± 95% CI* are the bootstrap percentiles. As fraction → 1.0 the score approaches the variant's expanded-pool number reported in `reports/phase2_summary.md`. As fraction → small, the score approaches what we would see under a 2025-official-style thin pool that does not match the variant's pick distribution.

## Support F1 — bootstrap mean and 95% CI per pool fraction

| Variant | 0.05 | 0.10 | 0.20 | 0.40 | 0.60 | 0.80 | 1.00 |
|---|---|---|---|---|---|---|---|
| phase1_baseline |  4.28 [ 2.54,  6.41] |  6.79 [ 4.89,  8.88] | 10.06 [ 7.83, 12.45] | 13.21 [11.63, 15.00] | 14.80 [13.33, 15.95] | 15.73 [14.81, 16.69] | 16.43 [16.43, 16.43] |
| allow_existing |  4.36 [ 2.59,  6.21] |  6.91 [ 4.95,  9.24] | 10.30 [ 8.09, 12.60] | 13.53 [11.86, 15.50] | 15.25 [13.86, 16.35] | 16.20 [15.21, 17.21] | 16.94 [16.94, 16.94] |
| no_rerank |  3.87 [ 2.48,  5.40] |  6.46 [ 4.88,  8.33] |  9.50 [ 7.70, 11.40] | 12.35 [10.76, 13.93] | 13.76 [12.57, 15.07] | 14.65 [13.85, 15.45] | 15.35 [15.35, 15.35] |
| bm25_rm3 |  2.34 [ 1.21,  3.83] |  3.77 [ 2.22,  5.59] |  5.67 [ 4.22,  7.17] |  7.36 [ 6.35,  8.56] |  8.14 [ 7.17,  9.02] |  8.59 [ 7.98,  9.18] |  8.97 [ 8.97,  8.97] |
| starter_baseline |  3.98 [ 2.32,  6.00] |  6.62 [ 4.77,  8.74] |  9.79 [ 7.52, 12.02] | 13.15 [11.58, 14.91] | 14.80 [13.57, 16.04] | 15.86 [14.98, 16.66] | 16.55 [16.55, 16.55] |

## Contradict F1 — bootstrap mean and 95% CI per pool fraction

| Variant | 0.05 | 0.10 | 0.20 | 0.40 | 0.60 | 0.80 | 1.00 |
|---|---|---|---|---|---|---|---|
| phase1_baseline |  0.97 [ 0.00,  2.27] |  1.80 [ 0.67,  3.14] |  3.43 [ 1.91,  4.76] |  6.13 [ 4.55,  8.05] |  8.31 [ 6.44, 10.12] | 10.32 [ 8.87, 11.62] | 12.01 [12.01, 12.01] |
| allow_existing |  0.97 [ 0.00,  2.27] |  1.80 [ 0.67,  3.14] |  3.43 [ 1.91,  4.76] |  6.13 [ 4.55,  8.05] |  8.31 [ 6.44, 10.12] | 10.32 [ 8.87, 11.62] | 12.01 [12.01, 12.01] |
| no_rerank |  0.97 [ 0.00,  2.11] |  1.78 [ 0.72,  3.11] |  3.37 [ 1.91,  4.71] |  6.02 [ 4.48,  7.81] |  8.17 [ 6.26,  9.99] | 10.12 [ 8.66, 11.40] | 11.75 [11.75, 11.75] |
| bm25_rm3 |  0.48 [ 0.00,  1.24] |  0.88 [ 0.21,  1.92] |  1.60 [ 0.64,  2.64] |  2.84 [ 1.80,  4.23] |  3.80 [ 2.89,  4.81] |  4.60 [ 3.76,  5.37] |  5.26 [ 5.26,  5.26] |
| starter_baseline |  0.40 [ 0.00,  1.03] |  0.77 [ 0.00,  1.75] |  1.50 [ 0.56,  2.54] |  2.84 [ 1.84,  3.99] |  3.79 [ 2.73,  5.01] |  4.66 [ 3.90,  5.55] |  5.34 [ 5.34,  5.34] |

## Visual: support F1 curve (ASCII)

Each bar is the bootstrap mean at one fraction, scaled to [0, 50 pp].

```
phase1_baseline             
  frac 0.05 | ██······················ |  4.28 pp
  frac 0.10 | ███····················· |  6.79 pp
  frac 0.20 | █████··················· | 10.06 pp
  frac 0.40 | ██████·················· | 13.21 pp
  frac 0.60 | ███████················· | 14.80 pp
  frac 0.80 | ████████················ | 15.73 pp
  frac 1.00 | ████████················ | 16.43 pp

allow_existing              
  frac 0.05 | ██······················ |  4.36 pp
  frac 0.10 | ███····················· |  6.91 pp
  frac 0.20 | █████··················· | 10.30 pp
  frac 0.40 | ██████·················· | 13.53 pp
  frac 0.60 | ███████················· | 15.25 pp
  frac 0.80 | ████████················ | 16.20 pp
  frac 1.00 | ████████················ | 16.94 pp

no_rerank                   
  frac 0.05 | ██······················ |  3.87 pp
  frac 0.10 | ███····················· |  6.46 pp
  frac 0.20 | █████··················· |  9.50 pp
  frac 0.40 | ██████·················· | 12.35 pp
  frac 0.60 | ███████················· | 13.76 pp
  frac 0.80 | ███████················· | 14.65 pp
  frac 1.00 | ███████················· | 15.35 pp

bm25_rm3                    
  frac 0.05 | █······················· |  2.34 pp
  frac 0.10 | ██······················ |  3.77 pp
  frac 0.20 | ███····················· |  5.67 pp
  frac 0.40 | ████···················· |  7.36 pp
  frac 0.60 | ████···················· |  8.14 pp
  frac 0.80 | ████···················· |  8.59 pp
  frac 1.00 | ████···················· |  8.97 pp

starter_baseline            
  frac 0.05 | ██······················ |  3.98 pp
  frac 0.10 | ███····················· |  6.62 pp
  frac 0.20 | █████··················· |  9.79 pp
  frac 0.40 | ██████·················· | 13.15 pp
  frac 0.60 | ███████················· | 14.80 pp
  frac 0.80 | ████████················ | 15.86 pp
  frac 1.00 | ████████················ | 16.55 pp

```

## Pool-bias delta — support F1 from frac=0.10 to frac=1.00

If a variant's score drops sharply when we shrink the pool, the variant is 'pool-bound' (its expanded-pool F1 is partly an artefact of how its picks overlap the full pool, not pure algorithmic quality). If a variant's score is approximately flat across fractions, the variant's score is *recall-bounded* (its expanded-pool F1 is roughly what its picks would score against any plausible pool slice).

| Variant | F1 @ frac=0.10 | F1 @ frac=1.00 | Δ (pp) |
|---|---|---|---|
| phase1_baseline |  6.79 | 16.43 | +9.64 |
| allow_existing |  6.91 | 16.94 | +10.03 |
| no_rerank |  6.46 | 15.35 | +8.89 |
| bm25_rm3 |  3.77 |  8.97 | +5.20 |
| starter_baseline |  6.62 | 16.55 | +9.93 |

## Interpretation

- Most pool-dependent variant on support: **allow_existing** (Δ +10.03 pp from 10% to 100% pool).
- Least pool-dependent variant on support: **bm25_rm3** (Δ +5.20 pp).

- Support F1 ranking at frac=1.00: allow_existing, starter_baseline, phase1_baseline, no_rerank, bm25_rm3
- Support F1 ranking at frac=0.10: allow_existing, phase1_baseline, starter_baseline, no_rerank, bm25_rm3

**Ranking instability**: the variant ordering changes between thin and full pool. This is the statistical fingerprint of pool-bias-driven F1 differences. Any leaderboard claim based on a thin pool (like the 2025 official) carries this same instability.