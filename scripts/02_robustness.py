"""
Inference robustness checks for Beaman et al. (2012).

Compares uncertainty under alternative inference procedures:
  1. Wild cluster bootstrap p-values (critical for 20-GP twice-reserved cell)
  2. Multiple testing corrections (Bonferroni, Benjamini-Hochberg)
  3. Alternative clustering (GP vs. village vs. household)
  4. Randomization inference / permutation tests
  5. Joint test across outcomes

These are standard referee concerns for any diff-in-diff with few clusters.

Outputs:
  - tabs/robustness_wild_bootstrap.md
  - tabs/robustness_multiple_testing.md
  - tabs/robustness_clustering.md
  - tabs/robustness_randomization.md
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from codebook import EDU_OUTCOMES, FORMULA, OUTCOMES, PARENT_OUTCOMES_LIST
from utils import (
    fit,
    hr,
    load_and_merge,
    wild_cluster_bootstrap,
    write_table,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Test 1: Wild cluster bootstrap
# ---------------------------------------------------------------------------


def test_wild_bootstrap(teen: pd.DataFrame, B: int = 999) -> None:
    """Wild cluster bootstrap for the twice-reserved gap difference."""
    hr(f"TEST: Wild Cluster Bootstrap (B={B})")
    print("\nWith only 20 GPs in the twice-reserved cell, asymptotic cluster")
    print("inference is unreliable; this is the recommended small-cluster fix.")
    print("(Cameron, Gelbach, & Miller 2008)")

    print(f"\n{'Outcome':<18}{'Coef':<10}{'SE_GP':<10}{'p (cluster)':<14}{'p (wild)':<14}")
    print("-" * 66)

    rows = []
    for y in OUTCOMES:
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m = fit(teen, y)
        coef = m.params["Twice:Female"]
        se = m.bse["Twice:Female"]
        p_cl = m.pvalues["Twice:Female"]

        _, p_wb, _ = wild_cluster_bootstrap(df, FORMULA.format(y=y), B=B)
        print(f"{y:<18}{coef:+.3f}     {se:.3f}     {p_cl:.3f}         {p_wb:.3f}")

        rows.append({
            "Outcome": y,
            "Coefficient": f"{coef:+.3f}",
            "SE (cluster)": f"{se:.3f}",
            "p (cluster)": f"{p_cl:.3f}",
            "p (wild bootstrap)": f"{p_wb:.3f}",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "robustness_wild_bootstrap", "Wild Cluster Bootstrap P-values")


# ---------------------------------------------------------------------------
# Test 2: Multiple testing correction
# ---------------------------------------------------------------------------


def test_multiple_testing(teen: pd.DataFrame) -> None:
    """Multiple testing correction for the four outcomes."""
    hr("TEST: Multiple Testing Correction")
    print("\nBonferroni: multiply p by number of tests")
    print("Benjamini-Hochberg: control false discovery rate")

    rs = []
    for y in OUTCOMES:
        m = fit(teen, y)
        rs.append((y, m.params["Twice:Female"], m.pvalues["Twice:Female"]))

    rs.sort(key=lambda x: x[2])
    ps = sorted(p for _, _, p in rs)

    print(f"\n{'Outcome':<18}{'Coef':<10}{'p (raw)':<10}{'p (Bonf)':<12}{'p (BH)':<10}")
    print("-" * 60)

    rows = []
    for y, b, p in rs:
        bonf = min(p * len(rs), 1.0)
        bh = min(p * len(rs) / (ps.index(p) + 1), 1.0)
        print(f"{y:<18}{b:+.3f}     {p:.3f}     {bonf:.3f}        {bh:.3f}")
        rows.append({
            "Outcome": y,
            "Coefficient": f"{b:+.3f}",
            "p (raw)": f"{p:.3f}",
            "p (Bonferroni)": f"{bonf:.3f}",
            "p (BH)": f"{bh:.3f}",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "robustness_multiple_testing", "Multiple Testing Corrections")


# ---------------------------------------------------------------------------
# Test 3: Alternative clustering
# ---------------------------------------------------------------------------


def test_clustering(teen: pd.DataFrame) -> None:
    """Sensitivity to clustering choice: GP vs. household vs. village."""
    hr("TEST: Clustering Choice (GP / Household / Village)")
    print("\nTreatment assigned at GP level, so GP clustering is correct.")
    print("HH and village shown for comparison.")

    print(f"\n{'Outcome':<18}{'Coef':<10}{'SE(GP)':<14}{'SE(HH)':<14}{'SE(Village)':<14}")
    print("-" * 70)

    rows = []
    for y in OUTCOMES:
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        f = FORMULA.format(y=y)
        m_gp = smf.ols(f, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["AA0_2b"]}
        )
        m_hh = smf.ols(f, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["hhserialid"]}
        )
        m_v = smf.ols(f, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["village_id"]}
        )

        coef = m_gp.params["Twice:Female"]
        print(
            f"{y:<18}{coef:+.3f}     {m_gp.bse['Twice:Female']:.3f}        "
            f"{m_hh.bse['Twice:Female']:.3f}        {m_v.bse['Twice:Female']:.3f}"
        )
        rows.append({
            "Outcome": y,
            "Coefficient": f"{coef:+.3f}",
            "SE (GP)": f"{m_gp.bse['Twice:Female']:.3f}",
            "SE (Household)": f"{m_hh.bse['Twice:Female']:.3f}",
            "SE (Village)": f"{m_v.bse['Twice:Female']:.3f}",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "robustness_clustering", "Clustering Sensitivity")


# ---------------------------------------------------------------------------
# Test 4: Randomization inference
# ---------------------------------------------------------------------------


def test_randomization_inference(
    teen: pd.DataFrame, B: int = 999, seed: int = 42
) -> None:
    """GP-level treatment permutation test."""
    hr(f"TEST: Randomization Inference (B={B})")
    print("\nPermute treatment assignment at GP level, preserving cell sizes.")

    gp_trt = teen[["AA0_2b", "twice_res", "once_res", "never_res"]].drop_duplicates(
        "AA0_2b"
    ).reset_index(drop=True)

    n_never = int(gp_trt["never_res"].sum())
    n_once = int(gp_trt["once_res"].sum())
    n_twice = int(gp_trt["twice_res"].sum())

    rng = np.random.default_rng(seed)

    print(f"\nGP cell sizes: Never={n_never}, Once={n_once}, Twice={n_twice}")
    print(f"\n{'Outcome':<18}{'Coef':<10}{'p (perm)':<12}{'5%-95% null'}")
    print("-" * 60)

    rows = []
    for y in OUTCOMES:
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m_obs = smf.ols(FORMULA.format(y=y), data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["AA0_2b"]}
        )
        obs = m_obs.params["Twice:Female"]

        gps = gp_trt["AA0_2b"].values
        null = []

        for _ in range(B):
            perm = rng.permutation(len(gps))
            once_p = np.zeros(len(gps))
            twice_p = np.zeros(len(gps))
            once_p[perm[n_never:n_never + n_once]] = 1
            twice_p[perm[n_never + n_once:]] = 1

            gp_perm = pd.DataFrame({"AA0_2b": gps, "_o": once_p, "_t": twice_p})
            dp = df.merge(gp_perm, on="AA0_2b", how="left").copy()
            dp["Once"] = dp["_o"]
            dp["Twice"] = dp["_t"]

            try:
                mp = smf.ols(FORMULA.format(y=y), data=dp).fit()
                null.append(mp.params["Twice:Female"])
            except Exception:
                continue

        null = np.asarray(null)
        p_perm = float((np.abs(null) >= np.abs(obs)).mean())
        lo, hi = np.percentile(null, 5), np.percentile(null, 95)

        print(f"{y:<18}{obs:+.3f}     {p_perm:.3f}      [{lo:+.3f}, {hi:+.3f}]")
        rows.append({
            "Outcome": y,
            "Coefficient": f"{obs:+.3f}",
            "p (permutation)": f"{p_perm:.3f}",
            "95% null CI": f"[{lo:+.3f}, {hi:+.3f}]",
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "robustness_randomization", "Randomization Inference")


# ---------------------------------------------------------------------------
# Test 5: Joint test across outcomes
# ---------------------------------------------------------------------------


def test_joint(teen: pd.DataFrame) -> None:
    """Stacked SUR-style estimation across all 4 outcomes."""
    hr("TEST: Joint Test (Stacked Estimation)")
    print("\nStack all 4 outcomes and estimate with outcome fixed effects.")

    df_long = []
    for y in OUTCOMES:
        sub = teen.dropna(subset=[y])[
            ["AA0_2b", "girl", "once_res", "twice_res", "hhserialid"]
        ].copy()
        sub["outcome_name"] = y
        sub["Y"] = teen.dropna(subset=[y])[y].values
        df_long.append(sub)

    df_long = pd.concat(df_long, ignore_index=True)
    df_long["Female"] = df_long["girl"]
    df_long["Once"] = df_long["once_res"]
    df_long["Twice"] = df_long["twice_res"]

    m = smf.ols(
        "Y ~ C(outcome_name) * (Once + Twice + Female + Once:Female + Twice:Female)",
        data=df_long,
    ).fit(cov_type="cluster", cov_kwds={"groups": df_long["AA0_2b"]})

    print(f"\nN observations (stacked across 4 outcomes): {len(df_long):,}")

    print("\nImplied Twice:Female effect at each outcome:")
    ref = sorted(df_long["outcome_name"].unique())[0]
    cov = m.cov_params()
    ref_eff = m.params["Twice:Female"]

    print(f"  {ref} (reference): {ref_eff:+.3f} ({m.bse['Twice:Female']:.3f})")

    for o in sorted(df_long["outcome_name"].unique()):
        if o == ref:
            continue
        inter = f"C(outcome_name)[T.{o}]:Twice:Female"
        if inter in m.params.index:
            b = ref_eff + m.params[inter]
            var_sum = (
                cov.loc["Twice:Female", "Twice:Female"]
                + cov.loc[inter, inter]
                + 2 * cov.loc["Twice:Female", inter]
            )
            se = float(np.sqrt(var_sum))
            print(f"  {o}: {b:+.3f} ({se:.3f})")


# ---------------------------------------------------------------------------
# Test 6: Wild bootstrap for ALL outcomes (Tables 1, 2, 3)
# ---------------------------------------------------------------------------


def _build_edu_outcomes(teen: pd.DataFrame) -> pd.DataFrame:
    """Build educational outcomes from household roster data.

    Column mappings from roster:
    - A1_8: Attending school/SSK/MSK/anganwadi (1=yes, 2=no)
    - A1_5: Can read and/or write (1=cannot, 2=read only, 3=read and write)
    - A1_7: Highest education achieved (grade level; 16=NA/not applicable)
    """
    teen = teen.copy()

    if "A1_8" in teen.columns:
        teen["attends_school"] = (teen["A1_8"] == 1).astype(int)
        teen.loc[teen["A1_8"].isna(), "attends_school"] = np.nan
    else:
        teen["attends_school"] = np.nan

    if "A1_5" in teen.columns:
        teen["can_read_write"] = (teen["A1_5"] == 3).astype(int)
        teen.loc[teen["A1_5"].isna(), "can_read_write"] = np.nan
    else:
        teen["can_read_write"] = np.nan

    if "A1_7" in teen.columns:
        teen["grade_completed"] = teen["A1_7"].replace([16, 99, 999], np.nan)
    else:
        teen["grade_completed"] = np.nan

    return teen


def _wild_bootstrap_simple(
    df: pd.DataFrame,
    formula: str,
    coef_name: str,
    cluster_col: str = "AA0_2b",
    B: int = 999,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Wild cluster bootstrap for a simple (non-interaction) coefficient."""
    use = df.copy()
    m_full = smf.ols(formula, data=use).fit(
        cov_type="cluster", cov_kwds={"groups": use[cluster_col]}
    )
    coef = m_full.params[coef_name]
    se = m_full.bse[coef_name]
    t_obs = coef / se
    p_cl = m_full.pvalues[coef_name]

    formula_null = formula.replace(f" + {coef_name}", "").replace(f"{coef_name} + ", "")
    if f"~ {coef_name}" in formula_null:
        formula_null = formula_null.replace(f"~ {coef_name}", "~ 1")
    m_null = smf.ols(formula_null, data=use).fit()
    fitted, resid = m_null.fittedvalues.values, m_null.resid.values

    rng = np.random.default_rng(seed)
    cl = use[cluster_col].values
    unique = np.unique(cl)
    boot = []
    y_name = formula.split("~")[0].strip()

    for _ in range(B):
        w = rng.choice([-1.0, 1.0], size=len(unique))
        cw = dict(zip(unique, w))
        wvec = np.fromiter((cw[c] for c in cl), dtype=float, count=len(cl))
        use["_y_b"] = fitted + wvec * resid
        try:
            mb = smf.ols(formula.replace(y_name, "_y_b"), data=use).fit(
                cov_type="cluster", cov_kwds={"groups": use[cluster_col]}
            )
            boot.append(mb.params[coef_name] / mb.bse[coef_name])
        except Exception:
            continue

    boot = np.asarray(boot)
    p_wb = float((np.abs(boot) >= np.abs(t_obs)).mean()) if len(boot) > 0 else np.nan
    return coef, se, p_cl, p_wb


def test_wild_bootstrap_all(
    teen: pd.DataFrame, adult: pd.DataFrame | None, B: int = 999
) -> None:
    """Wild cluster bootstrap for ALL 10 primary outcomes across Tables 1-3."""
    hr(f"TEST: Wild Cluster Bootstrap - ALL Outcomes (B={B})")
    print("\nCovers all 10 primary hypotheses:")
    print("  - Table 1: Parents' aspirations (3 outcomes, 'Twice' coef, women-only)")
    print("  - Table 2: Adolescents' aspirations (4 outcomes, 'Twice:Female' coef)")
    print("  - Table 3: Educational outcomes (3 outcomes, 'Twice:Female' coef)")

    rows = []

    teen = _build_edu_outcomes(teen)

    print(f"\n{'Table':<8}{'Outcome':<28}{'Coef':<10}{'SE':<10}{'p(cluster)':<12}{'p(wild)':<10}")
    print("-" * 88)

    for y in OUTCOMES:
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m = fit(teen, y)
        coef = m.params["Twice:Female"]
        se = m.bse["Twice:Female"]
        p_cl = m.pvalues["Twice:Female"]

        _, p_wb, _ = wild_cluster_bootstrap(df, FORMULA.format(y=y), B=B)
        print(f"{'2':<8}{y:<28}{coef:+.4f}    {se:.4f}    {p_cl:.4f}       {p_wb:.4f}")

        rows.append({
            "Table": "2",
            "Outcome": y,
            "Coefficient": coef,
            "SE": se,
            "p (cluster)": p_cl,
            "p (wild)": p_wb,
            "Coef_name": "Twice:Female",
        })

    for y in EDU_OUTCOMES:
        if y not in teen.columns or teen[y].isna().all():
            continue
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m = smf.ols(FORMULA.format(y=y), data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["AA0_2b"]}
        )
        coef = m.params["Twice:Female"]
        se = m.bse["Twice:Female"]
        p_cl = m.pvalues["Twice:Female"]

        _, p_wb, _ = wild_cluster_bootstrap(df, FORMULA.format(y=y), B=B)
        print(f"{'3':<8}{y:<28}{coef:+.4f}    {se:.4f}    {p_cl:.4f}       {p_wb:.4f}")

        rows.append({
            "Table": "3",
            "Outcome": y,
            "Coefficient": coef,
            "SE": se,
            "p (cluster)": p_cl,
            "p (wild)": p_wb,
            "Coef_name": "Twice:Female",
        })

    if adult is not None:
        women = adult[adult["Female"] == 1].copy()
        women["Once"] = women["once_res"]
        women["Twice"] = women["twice_res"]

        for y in PARENT_OUTCOMES_LIST:
            if y not in women.columns or women[y].isna().all():
                continue
            use = women.dropna(subset=[y]).copy()
            if len(use) < 10:
                continue

            formula = f"{y} ~ Once + Twice"
            m = smf.ols(formula, data=use).fit(
                cov_type="cluster", cov_kwds={"groups": use["AA0_2b"]}
            )
            coef = m.params["Twice"]
            se = m.bse["Twice"]
            p_cl = m.pvalues["Twice"]

            _, _, _, p_wb = _wild_bootstrap_simple(
                use, formula, "Twice", B=B
            )
            print(f"{'1':<8}{y:<28}{coef:+.4f}    {se:.4f}    {p_cl:.4f}       {p_wb:.4f}")

            rows.append({
                "Table": "1",
                "Outcome": y,
                "Coefficient": coef,
                "SE": se,
                "p (cluster)": p_cl,
                "p (wild)": p_wb,
                "Coef_name": "Twice",
            })

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["Table", "Outcome"]).reset_index(drop=True)

    df_display = df_out[["Table", "Outcome", "Coefficient", "SE", "p (cluster)", "p (wild)"]].copy()
    df_display["Coefficient"] = df_display["Coefficient"].apply(lambda x: f"{x:+.4f}")
    df_display["SE"] = df_display["SE"].apply(lambda x: f"{x:.4f}")
    df_display["p (cluster)"] = df_display["p (cluster)"].apply(lambda x: f"{x:.4f}")
    df_display["p (wild)"] = df_display["p (wild)"].apply(lambda x: f"{x:.4f}")

    write_table(
        df_display, "robustness_wild_bootstrap_all", "Wild Cluster Bootstrap - All Outcomes"
    )

    return df_out


# ---------------------------------------------------------------------------
# Test 7: Multiple testing correction for ALL outcomes
# ---------------------------------------------------------------------------


def test_multiple_testing_all(
    teen: pd.DataFrame, adult: pd.DataFrame | None
) -> None:
    """Multiple testing correction for all 10 primary outcomes."""
    hr("TEST: Multiple Testing Correction - ALL Outcomes")
    print("\nCollects p-values from Tables 1, 2, 3 and applies corrections:")
    print("  - Bonferroni: p × k (conservative, controls FWER)")
    print("  - Holm (step-down): ordered adjustments (less conservative)")
    print("  - Benjamini-Hochberg: FDR control (least conservative)")

    teen = _build_edu_outcomes(teen)
    results = []

    for y in OUTCOMES:
        m = fit(teen, y)
        results.append({
            "Table": "2",
            "Outcome": y,
            "Coef": m.params["Twice:Female"],
            "p_raw": m.pvalues["Twice:Female"],
        })

    for y in EDU_OUTCOMES:
        if y not in teen.columns or teen[y].isna().all():
            continue
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m = smf.ols(FORMULA.format(y=y), data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["AA0_2b"]}
        )
        results.append({
            "Table": "3",
            "Outcome": y,
            "Coef": m.params["Twice:Female"],
            "p_raw": m.pvalues["Twice:Female"],
        })

    if adult is not None:
        women = adult[adult["Female"] == 1].copy()
        women["Once"] = women["once_res"]
        women["Twice"] = women["twice_res"]

        for y in PARENT_OUTCOMES_LIST:
            if y not in women.columns or women[y].isna().all():
                continue
            use = women.dropna(subset=[y]).copy()
            if len(use) < 10:
                continue

            formula = f"{y} ~ Once + Twice"
            m = smf.ols(formula, data=use).fit(
                cov_type="cluster", cov_kwds={"groups": use["AA0_2b"]}
            )
            results.append({
                "Table": "1",
                "Outcome": y,
                "Coef": m.params["Twice"],
                "p_raw": m.pvalues["Twice"],
            })

    k = len(results)
    results_sorted = sorted(results, key=lambda x: x["p_raw"])

    for i, r in enumerate(results_sorted):
        r["rank"] = i + 1
        r["p_bonf"] = min(r["p_raw"] * k, 1.0)
        r["p_holm"] = min(r["p_raw"] * (k - i), 1.0)
        r["p_bh"] = min(r["p_raw"] * k / (i + 1), 1.0)

    for i in range(1, len(results_sorted)):
        results_sorted[i]["p_holm"] = max(
            results_sorted[i]["p_holm"], results_sorted[i - 1]["p_holm"]
        )

    for i in range(len(results_sorted) - 2, -1, -1):
        results_sorted[i]["p_bh"] = min(
            results_sorted[i]["p_bh"], results_sorted[i + 1]["p_bh"]
        )

    print(f"\nTotal hypotheses tested: {k}")
    print(f"\n{'Table':<6}{'Outcome':<28}{'Coef':<10}{'p(raw)':<10}"
          f"{'p(Bonf)':<10}{'p(Holm)':<10}{'p(BH)':<10}")
    print("-" * 94)

    for r in results_sorted:
        sig_raw = "*" if r["p_raw"] < 0.05 else ""
        sig_bonf = "*" if r["p_bonf"] < 0.05 else ""
        sig_holm = "*" if r["p_holm"] < 0.05 else ""
        sig_bh = "*" if r["p_bh"] < 0.05 else ""
        print(
            f"{r['Table']:<6}{r['Outcome']:<28}{r['Coef']:+.4f}    "
            f"{r['p_raw']:.4f}{sig_raw:<3} {r['p_bonf']:.4f}{sig_bonf:<3} "
            f"{r['p_holm']:.4f}{sig_holm:<3} {r['p_bh']:.4f}{sig_bh:<3}"
        )

    rows = []
    for r in results_sorted:
        sig_raw = "*" if r["p_raw"] < 0.05 else ""
        sig_bonf = "*" if r["p_bonf"] < 0.05 else ""
        sig_holm = "*" if r["p_holm"] < 0.05 else ""
        sig_bh = "*" if r["p_bh"] < 0.05 else ""
        rows.append({
            "Table": r["Table"],
            "Outcome": r["Outcome"],
            "Coefficient": f"{r['Coef']:+.4f}",
            "p (raw)": f"{r['p_raw']:.4f}{sig_raw}",
            "p (Bonferroni)": f"{r['p_bonf']:.4f}{sig_bonf}",
            "p (Holm)": f"{r['p_holm']:.4f}{sig_holm}",
            "p (BH)": f"{r['p_bh']:.4f}{sig_bh}",
        })

    df_out = pd.DataFrame(rows)
    write_table(
        df_out, "robustness_multiple_testing_all",
        "Multiple Testing Corrections - All Outcomes"
    )

    return results_sorted


# ---------------------------------------------------------------------------
# Test 8: Inference summary dashboard
# ---------------------------------------------------------------------------


def test_inference_summary(
    teen: pd.DataFrame, adult: pd.DataFrame | None, B: int = 999
) -> None:
    """Summary dashboard combining wild bootstrap and multiple testing."""
    hr("INFERENCE SUMMARY DASHBOARD")
    print("\nComprehensive inference for all 10 primary hypotheses")

    teen = _build_edu_outcomes(teen)
    results = []

    for y in OUTCOMES:
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m = fit(teen, y)
        coef = m.params["Twice:Female"]
        se = m.bse["Twice:Female"]
        p_cl = m.pvalues["Twice:Female"]
        _, p_wb, _ = wild_cluster_bootstrap(df, FORMULA.format(y=y), B=B)

        results.append({
            "Table": "2",
            "Outcome": y,
            "Coef": coef,
            "SE": se,
            "p_raw": p_cl,
            "p_wild": p_wb,
        })

    for y in EDU_OUTCOMES:
        if y not in teen.columns or teen[y].isna().all():
            continue
        df = teen.dropna(subset=[y]).copy()
        df["Female"] = df["girl"]
        df["Once"] = df["once_res"]
        df["Twice"] = df["twice_res"]

        m = smf.ols(FORMULA.format(y=y), data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["AA0_2b"]}
        )
        coef = m.params["Twice:Female"]
        se = m.bse["Twice:Female"]
        p_cl = m.pvalues["Twice:Female"]
        _, p_wb, _ = wild_cluster_bootstrap(df, FORMULA.format(y=y), B=B)

        results.append({
            "Table": "3",
            "Outcome": y,
            "Coef": coef,
            "SE": se,
            "p_raw": p_cl,
            "p_wild": p_wb,
        })

    if adult is not None:
        women = adult[adult["Female"] == 1].copy()
        women["Once"] = women["once_res"]
        women["Twice"] = women["twice_res"]

        for y in PARENT_OUTCOMES_LIST:
            if y not in women.columns or women[y].isna().all():
                continue
            use = women.dropna(subset=[y]).copy()
            if len(use) < 10:
                continue

            formula = f"{y} ~ Once + Twice"
            m = smf.ols(formula, data=use).fit(
                cov_type="cluster", cov_kwds={"groups": use["AA0_2b"]}
            )
            coef = m.params["Twice"]
            se = m.bse["Twice"]
            p_cl = m.pvalues["Twice"]

            _, _, _, p_wb = _wild_bootstrap_simple(use, formula, "Twice", B=B)

            results.append({
                "Table": "1",
                "Outcome": y,
                "Coef": coef,
                "SE": se,
                "p_raw": p_cl,
                "p_wild": p_wb,
            })

    k = len(results)
    results_sorted = sorted(results, key=lambda x: x["p_raw"])

    for i, r in enumerate(results_sorted):
        r["p_bonf"] = min(r["p_raw"] * k, 1.0)
        r["p_bh"] = min(r["p_raw"] * k / (i + 1), 1.0)

    for i in range(len(results_sorted) - 2, -1, -1):
        results_sorted[i]["p_bh"] = min(
            results_sorted[i]["p_bh"], results_sorted[i + 1]["p_bh"]
        )

    results_by_table = sorted(results_sorted, key=lambda x: (x["Table"], x["Outcome"]))

    print(f"\n{'Table':<6}{'Outcome':<28}{'Coef':<10}{'p(raw)':<10}{'p(wild)':<10}"
          f"{'p(Bonf)':<10}{'p(BH)':<10}{'Sig?':<8}")
    print("-" * 102)

    for r in results_by_table:
        sig_any = r["p_raw"] < 0.05
        sig_wild = r["p_wild"] < 0.05
        sig_bonf = r["p_bonf"] < 0.05
        sig_bh = r["p_bh"] < 0.05

        if sig_bonf:
            sig_str = "***"
        elif sig_bh:
            sig_str = "**"
        elif sig_wild:
            sig_str = "*"
        elif sig_any:
            sig_str = "."
        else:
            sig_str = ""

        print(
            f"{r['Table']:<6}{r['Outcome']:<28}{r['Coef']:+.4f}    "
            f"{r['p_raw']:.4f}    {r['p_wild']:.4f}    {r['p_bonf']:.4f}    "
            f"{r['p_bh']:.4f}    {sig_str:<8}"
        )

    print("\nSignificance codes: *** = Bonferroni, ** = BH, * = Wild bootstrap, . = raw only")

    rows = []
    for r in results_by_table:
        sig_bonf = r["p_bonf"] < 0.05
        sig_bh = r["p_bh"] < 0.05
        sig_wild = r["p_wild"] < 0.05
        sig_raw = r["p_raw"] < 0.05

        if sig_bonf:
            sig_str = "***"
        elif sig_bh:
            sig_str = "**"
        elif sig_wild:
            sig_str = "*"
        elif sig_raw:
            sig_str = "."
        else:
            sig_str = ""

        rows.append({
            "Table": r["Table"],
            "Outcome": r["Outcome"],
            "Coefficient": f"{r['Coef']:+.4f}",
            "p (raw)": f"{r['p_raw']:.4f}",
            "p (wild)": f"{r['p_wild']:.4f}",
            "p (Bonferroni)": f"{r['p_bonf']:.4f}",
            "p (BH)": f"{r['p_bh']:.4f}",
            "Sig": sig_str,
        })

    df_out = pd.DataFrame(rows)
    write_table(df_out, "inference_summary", "Inference Summary Dashboard")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


ALL_TESTS = {
    1: ("wild-bootstrap", test_wild_bootstrap, "teen-B"),
    2: ("multiple-testing", test_multiple_testing, "teen"),
    3: ("clustering", test_clustering, "teen"),
    4: ("randomization-inf", test_randomization_inference, "teen-B"),
    5: ("joint-test", test_joint, "teen"),
    6: ("wild-bootstrap-all", test_wild_bootstrap_all, "both-B"),
    7: ("multiple-testing-all", test_multiple_testing_all, "both"),
    8: ("inference-summary", test_inference_summary, "both-B"),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the .dta files.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=999,
        help="Number of bootstrap and permutation draws.",
    )
    parser.add_argument(
        "--tests",
        type=int,
        nargs="*",
        help=f"Subset of tests to run (1-{len(ALL_TESTS)}). Default: all.",
    )
    args = parser.parse_args()

    tests_to_run = args.tests if args.tests else list(ALL_TESTS.keys())

    needs_adult = any(
        ALL_TESTS[k][2].startswith("both") for k in tests_to_run if k in ALL_TESTS
    )
    teen, adult = load_and_merge(args.data_dir, include_adult=needs_adult)

    for k in tests_to_run:
        if k not in ALL_TESTS:
            continue
        _, fn, kind = ALL_TESTS[k]
        if kind == "teen":
            fn(teen)
        elif kind == "teen-B":
            fn(teen, B=args.bootstrap_reps)
        elif kind == "both":
            fn(teen, adult)
        elif kind == "both-B":
            fn(teen, adult, B=args.bootstrap_reps)


if __name__ == "__main__":
    main()
