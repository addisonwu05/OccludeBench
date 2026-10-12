"""
REBUTTAL: does human accuracy improve over the 6 questions each Prolific
participant answered? (learning / familiarity effect vs. stagnation)

Position (1..6) is derived from timestamp order within each participant.
Correctness = consensus_correct from the judged human results.

statsmodels is broken against the installed scipy, so the logistic regression
and its cluster-robust (by participant) standard errors are implemented by hand
via IRLS + a sandwich estimator.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

# ---- Load & join -----------------------------------------------------------
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
base = json.load(open(os.path.join(ROOT, "data/occlusion_prolific_human_baseline.json")))
judged = json.load(open(os.path.join(ROOT, "data/eval_results_human/human_prolific_judged_results.json")))

ts = {(r["prolific_pid"], r["image_id"]): r["timestamp"] for r in base}

rows = []
for r in judged:
    key = (r["prolific_pid"], r["image_id"])
    if key not in ts or r.get("consensus_correct") is None:
        continue
    rows.append({
        "pid": r["prolific_pid"],
        "image_id": r["image_id"],
        "timestamp": ts[key],
        "correct": int(bool(r["consensus_correct"])),
        "rt_ms": r.get("response_time_ms"),
    })

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["pid", "timestamp"])
df["position"] = df.groupby("pid").cumcount() + 1
df = df[df["position"] <= 6]

print(f"Joined responses: {len(df)}  | participants: {df['pid'].nunique()}")

# ---- Accuracy by position --------------------------------------------------
print("\n=== Accuracy by question position ===")
g = df.groupby("position")["correct"].agg(["mean", "sum", "count"])
g["se"] = np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
for pos, row in g.iterrows():
    lo = row["mean"] - 1.96 * row["se"]
    hi = row["mean"] + 1.96 * row["se"]
    print(f"  Q{pos}: acc={row['mean']:.3f}  ({int(row['sum'])}/{int(row['count'])})  "
          f"95% CI [{lo:.3f}, {hi:.3f}]")

# ---- Test 1: linear trend --------------------------------------------------
print("\n=== Test 1: Linear trend across positions ===")
r_pb, p_pb = stats.pointbiserialr(df["correct"], df["position"])
print(f"  Point-biserial r(correct, position) = {r_pb:.4f}, p = {p_pb:.4f}")

counts = df.groupby("position")["correct"].agg(["sum", "count"])
n_i = counts["count"].values.astype(float)
x_i = counts.index.values.astype(float)
p1_i = counts["sum"].values.astype(float)
N = n_i.sum(); R = p1_i.sum(); pbar = R / N
xbar = (n_i * x_i).sum() / N
num = ((p1_i - n_i * pbar) * (x_i - xbar)).sum()
den = pbar * (1 - pbar) * (n_i * (x_i - xbar) ** 2).sum()
z_ca = num / np.sqrt(den)
p_ca = 2 * (1 - stats.norm.cdf(abs(z_ca)))
print(f"  Cochran-Armitage trend: z = {z_ca:.3f}, p = {p_ca:.4f}")

# Q1 vs Q6 two-proportion z-test (first vs last)
def two_prop_z(s1, n1, s2, n2):
    p1, p2 = s1 / n1, s2 / n2
    pp = (s1 + s2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))
z16, p16 = two_prop_z(counts.loc[1, "sum"], counts.loc[1, "count"],
                      counts.loc[6, "sum"], counts.loc[6, "count"])
print(f"  Q1 vs Q6 two-proportion z = {z16:.3f}, p = {p16:.4f}  "
      f"(Q1={g.loc[1,'mean']:.3f} vs Q6={g.loc[6,'mean']:.3f})")

# ---- Test 2: logistic regression, cluster-robust SE by participant ---------
print("\n=== Test 2: Logistic  correct ~ position  (cluster-robust SE by pid) ===")
def logit_irls(X, y, iters=100, tol=1e-10):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        WX = X * W[:, None]
        H = X.T @ WX
        grad = X.T @ (y - mu)
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta, mu

y = df["correct"].values.astype(float)
X = np.column_stack([np.ones(len(df)), df["position"].values.astype(float)])
beta, mu = logit_irls(X, y)

# cluster-robust (sandwich) covariance, clustered by participant
W = mu * (1 - mu)
bread = np.linalg.inv((X * W[:, None]).T @ X)
u = X * (y - mu)[:, None]  # score contributions
meat = np.zeros((X.shape[1], X.shape[1]))
pid_arr = df["pid"].values
for pid in np.unique(pid_arr):
    idx = pid_arr == pid
    ug = u[idx].sum(axis=0)
    meat += np.outer(ug, ug)
cov = bread @ meat @ bread
se = np.sqrt(np.diag(cov))
z = beta / se
pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
print(f"  intercept: coef={beta[0]:.4f}  se={se[0]:.4f}")
print(f"  position : coef={beta[1]:.4f}  se={se[1]:.4f}  OR={np.exp(beta[1]):.4f}  "
      f"z={z[1]:.3f}  p={pvals[1]:.4f}")
print(f"  -> OR>1 = improvement over questions; OR≈1 & p>.05 = stagnation")

# ---- Secondary: response time by position ----------------------------------
print("\n=== Secondary: response time by position (speed-up = familiarity) ===")
rtdf = df.dropna(subset=["rt_ms"])
for pos, v in rtdf.groupby("position")["rt_ms"].median().items():
    print(f"  Q{pos}: median RT = {v/1000:.1f}s")
r_rt, p_rt = stats.spearmanr(rtdf["position"], rtdf["rt_ms"])
print(f"  Spearman(position, RT) = {r_rt:.4f}, p = {p_rt:.4g}")
