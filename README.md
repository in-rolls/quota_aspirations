# Women Leaders and Adolescent Aspirations

Python replication and robustness analysis of:

> Beaman, L., Duflo, E., Pande, R., & Topalova, P. (2012).
> "Female Leadership Raises Aspirations and Educational Attainment for Girls:
> A Policy Experiment in India." *Science*, 335(6068), 582-586.
> [doi:10.1126/science.1212382](https://doi.org/10.1126/science.1212382)

## Overview

This repository replicates the main tables from Beaman et al. (2012) and conducts inference robustness checks. The paper examines whether exposure to female political leaders affects adolescent girls' aspirations in rural India, exploiting random assignment of village council seats to women.

## Comparison-group means

The table compares published and reproduced means in never-reserved villages. Differences vary by outcome; several exceed rounding tolerance.

| Outcome | Boys (paper) | Boys (reproduced) | Girls (paper) | Girls (reproduced) |
|---------|--------------|-------------------|---------------|---------------------|
| no_housewife | 0.998 | 0.998 | 0.600 | 0.585 |
| wish_graduate | 0.296 | 0.290 | 0.195 | 0.190 |
| marry_after18 | 0.980 | 0.980 | 0.660 | 0.662 |
| wish_pradhan | 0.499 | 0.504 | 0.485 | 0.487 |

## Estimates and uncertainty

P-values for the twice-reserved × female interaction across all seven outcomes:

| Table | Outcome | Coefficient | p (raw) | p (wild) | p (Bonferroni) | p (BH) |
|------:|:--------|------------:|--------:|---------:|---------------:|-------:|
| 2 | no_housewife | 0.089 | 0.036 | 0.051 | 0.252 | 0.160 |
| 2 | marry_after18 | 0.089 | 0.069 | 0.131 | 0.480 | 0.160 |
| 3 | can_read_write | 0.062 | 0.069 | 0.101 | 0.479 | 0.160 |
| 2 | wish_graduate | 0.018 | 0.698 | 0.667 | 1.000 | 0.801 |
| 2 | wish_pradhan | -0.014 | 0.801 | 0.768 | 1.000 | 0.801 |
| 3 | attends_school | 0.023 | 0.754 | 0.879 | 1.000 | 0.801 |
| 3 | grade_completed | 0.153 | 0.485 | 0.434 | 1.000 | 0.801 |

**Notes on inference corrections:**
- **Wild cluster bootstrap** addresses the small number of clusters (20 GPs in the twice-reserved cell) using Cameron, Gelbach & Miller (2008).
- **Bonferroni** controls family-wise error rate across seven tests.
- **BH** (Benjamini-Hochberg) controls false discovery rate.

One outcome (no_housewife) shows p < 0.05 with cluster-robust standard errors. After wild bootstrap correction, p = 0.051. After Bonferroni correction across all seven outcomes, no outcome reaches p < 0.05.

## Data

**Sources:**
- [Harvard Dataverse (J-PAL)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O3UKFO)
- [Stanford Redivis](https://stanford.redivis.com/datasets/q707-5rwxymgw0/tables)

**Required files:**

| File | Rows | Description |
|------|------|-------------|
| `powerful_women_in_india_teenager_survey.dta` | 3,680 | Adolescent aspirations |
| `powerful_women_in_india_pradhan_seats_reserved_for_women.dta` | 165 | Treatment assignment |
| `powerful_women_in_india_household_roster.dta` | 37,263 | Demographics |
| `powerful_women_in_india_household_survey.dta` | 7,425 | Location data |
| `powerful_women_in_india_adult_survey.dta` | 13,508 | Adult attitudes |

**Treatment cells:**
- Twice-reserved: 20 GPs (both cycles)
- Once-reserved: 70 GPs (one cycle)
- Never-reserved: 72 GPs (control)

## Methods

### Robustness Checks

| Method | Purpose | Reference |
|--------|---------|-----------|
| Wild cluster bootstrap | Small-sample cluster inference | Cameron, Gelbach & Miller (2008) |
| Bonferroni / Holm | Family-wise error rate control | |
| Benjamini-Hochberg | False discovery rate control | |
| Randomization inference | Permutation-based p-values | |
| Alternative clustering | GP, household, village levels | |

### Sensitivity Analysis

| Analysis | Purpose |
|----------|---------|
| Leave-one-out | Influence of individual GPs |
| Ceiling effects | Boys near 100% on some outcomes |
| Age placebo | Adults socialized pre-1993 |
| Unused survey items | E_1 to E_10 gender attitudes |
| Time use | School attendance, domestic chores |

## Running the Code

```bash
pip install -r requirements.txt

python scripts/01_replicate.py --data-dir data
python scripts/02_robustness.py --data-dir data
python scripts/03_mechanisms.py --data-dir data
python scripts/04_sensitivity.py --data-dir data
```

## Output Files

| File | Description |
|------|-------------|
| `tabs/inference_summary.md` | Combined p-values across corrections |
| `tabs/table2_baselines.md` | Aspiration baselines |
| `tabs/table2_coefficients.md` | Aspiration treatment effects |
| `tabs/table3_baselines.md` | Education baselines |
| `tabs/table3_education.md` | Education treatment effects |
| `tabs/robustness_wild_bootstrap_all.md` | Wild bootstrap for all outcomes |
| `tabs/robustness_multiple_testing_all.md` | Multiple testing corrections |
| `tabs/robustness_clustering.md` | Alternative clustering |
| `tabs/robustness_randomization.md` | Randomization inference |

## Codebook

Variable definitions are in [scripts/codebook.py](scripts/codebook.py).

**Key variables:**

| Variable | Source | Coding |
|----------|--------|--------|
| `no_housewife` | `B1_3` | 1 if occupation ≠ housewife |
| `wish_graduate` | `B1_1` | 1 if `B1_1 == 13` |
| `marry_after18` | `B1_2` | 1 if `B1_2 >= 19` |
| `wish_pradhan` | `B1_4` | 1 if `B1_4 == 1` |
| `twice_res` | Treatment | 1 if reserved both cycles |

## Caveats

- 162 of 165 GPs match between treatment and household files
- Wild bootstrap uses Rademacher weights imposing the null
- Bootstrap p-values shift slightly across runs; use `--bootstrap-reps 4999` for stability

## License

MIT
