"""Statistical engine for the Salon Experiment & Causal Analysis app.
Only needs pandas, numpy and scipy (no statsmodels), so deployment stays light."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
ROOT = HERE.parent
CUTOFF = "2026-07"  # intervention month


def _find(name):
    for cand in (name, name + ".gz"):
        for folder in (HERE, HERE / "data"):
            if (folder / cand).exists():
                return folder / cand
        hits = list(ROOT.rglob(cand))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found anywhere in the repo")

def load_experiment():
    return pd.read_csv(_find("experiment_cohort_8000.csv"))


def load_outlet_month():
    cols = ["appointment_id", "appointment_date", "outlet_id", "region",
            "no_show_flag", "realized_revenue_inr"]
    a = pd.read_csv(_find("appointments_300k.csv"), usecols=cols, parse_dates=["appointment_date"])
    a["month"] = a["appointment_date"].dt.to_period("M").astype(str)
    om = a.groupby(["outlet_id", "region", "month"], as_index=False).agg(
        n=("appointment_id", "size"),
        revenue=("realized_revenue_inr", "mean"),
        no_show=("no_show_flag", "mean"))
    om["treated"] = (om["region"] == "Treatment Region").astype(int)
    om["post"] = (om["month"] >= CUTOFF).astype(int)
    summary = dict(rows=len(a), outlets=a["outlet_id"].nunique())
    return om, summary


# ---------- A/B design ----------
def sample_size(base, target, alpha=0.05, power=0.80):
    """Clients per arm for a two-sided two-proportion test (Cohen's h, 1:1)."""
    h = abs(2 * np.arcsin(np.sqrt(target)) - 2 * np.arcsin(np.sqrt(base)))
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(2 * (za + zb) ** 2 / h ** 2))


# ---------- A/B analysis ----------
def split(exp):
    return exp[exp.experiment_group == "Treatment"], exp[exp.experiment_group == "Control"]


def srm_p(exp):
    t, c = split(exp)
    return stats.binomtest(len(t), len(t) + len(c), 0.5).pvalue


def balance(exp):
    t, c = split(exp)
    rows = []
    for col in ["age", "prior_visits_90d", "prior_revenue_90d_inr", "new_client_flag"]:
        p = stats.ttest_ind(t[col], c[col], equal_var=False).pvalue
        rows.append((col, t[col].mean(), c[col].mean(), p))
    return pd.DataFrame(rows, columns=["Covariate", "Treatment", "Control", "p-value"])


def welch_ci(x, y, alpha=0.05):
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    d = x.mean() - y.mean()
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    crit = stats.t.ppf(1 - alpha / 2, df)
    se = np.sqrt(vx + vy)
    return d, d - crit * se, d + crit * se, 2 * stats.t.sf(abs(d / se), df)


def bootstrap_ci(x, y, B=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    diffs = (rng.choice(x, (B, len(x))).mean(1) - rng.choice(y, (B, len(y))).mean(1))
    return x.mean() - y.mean(), *np.quantile(diffs, [alpha / 2, 1 - alpha / 2])


def segments(exp):
    e = exp.copy()
    e["segment"] = pd.cut(e.prior_visits_90d, [-1, 0, 2, 5, 100],
                          labels=["New (0)", "Low (1-2)", "Medium (3-5)", "High (6+)"])
    out = []
    for seg, g in e.groupby("segment", observed=True):
        t, c = split(g)
        d, lo, hi, _ = welch_ci(t.retained_within_60d, c.retained_within_60d)
        out.append((str(seg), len(g), t.retained_within_60d.mean(), c.retained_within_60d.mean(), d, lo, hi))
    return pd.DataFrame(out, columns=["Segment", "N", "Treatment", "Control", "Diff", "Lo", "Hi"])


def adjusted_effect(exp):
    """Regression adjustment (ANCOVA-style) with HC3 standard errors."""
    y = exp.retained_within_60d.to_numpy(float)
    D = (exp.experiment_group == "Treatment").to_numpy(float)
    X = np.column_stack([np.ones(len(exp)), D, exp.prior_visits_90d, exp.prior_revenue_90d_inr, exp.age])
    X0 = np.column_stack([np.ones(len(exp)), D])

    def fit(X):
        XtXi = np.linalg.inv(X.T @ X)
        b = XtXi @ X.T @ y
        u = y - X @ b
        h = np.einsum("ij,jk,ik->i", X, XtXi, X)
        meat = (X * (u / (1 - h))[:, None]).T @ (X * (u / (1 - h))[:, None])
        se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
        return b[1], se[1], u.var()

    b1, se1, v1 = fit(X0)
    b2, se2, v2 = fit(X)
    z = stats.norm.ppf(0.975)
    return dict(raw=b1, raw_se=se1, adj=b2, adj_se=se2, lo=b2 - z * se2, hi=b2 + z * se2,
                var_reduction=1 - v2 / v1, n_ratio=(se2 / se1) ** 2)


# ---------- Difference-in-differences ----------
def _wls_cluster(y, X, w, groups, k_idx):
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    XtXi = np.linalg.pinv(Xw.T @ Xw)
    b = XtXi @ Xw.T @ yw
    u = yw - Xw @ b
    ids = pd.factorize(groups)[0]
    G, N, k = ids.max() + 1, len(y), np.linalg.matrix_rank(Xw)
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, ids, Xw * u[:, None])
    V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((N - 1) / (N - k))
    se = np.sqrt(V[k_idx, k_idx])
    z = stats.norm.ppf(0.975)
    return b[k_idx], b[k_idx] - z * se, b[k_idx] + z * se, 2 * stats.norm.sf(abs(b[k_idx] / se))


def did(om, outcome="revenue"):
    d = om.copy()
    d["did"] = d.treated * d.post
    X = pd.concat([d[["did"]], pd.get_dummies(d.outlet_id, drop_first=True, dtype=float),
                   pd.get_dummies(d.month, drop_first=True, dtype=float)], axis=1)
    X.insert(0, "const", 1.0)
    return _wls_cluster(d[outcome].to_numpy(float), X.to_numpy(float), d.n.to_numpy(float),
                        d.outlet_id.to_numpy(), 1)


def pretrend(om, outcome="revenue"):
    d = om[om.month < CUTOFF].copy()
    d["t"] = pd.PeriodIndex(d.month, freq="M").astype(int)
    d["inter"] = d.treated * d.t
    X = pd.concat([d[["inter", "t"]], pd.get_dummies(d.outlet_id, drop_first=True, dtype=float)], axis=1)
    X.insert(0, "const", 1.0)
    est, lo, hi, p = _wls_cluster(d[outcome].to_numpy(float), X.to_numpy(float), d.n.to_numpy(float),
                                  d.outlet_id.to_numpy(), 1)
    return est, p


def region_trend(om, outcome="revenue"):
    g = om.groupby(["region", "month"]).apply(
        lambda x: np.average(x[outcome], weights=x.n), include_groups=False).rename("value").reset_index()
    return g
