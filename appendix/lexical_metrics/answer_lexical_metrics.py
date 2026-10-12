"""
REBUTTAL: are the ground-truth answers for DIRECT (reflection/shadow) images
really more "niche and abstract" than the INFERENTIAL ones?

We quantify each answer string with a SUITE of objective lexical norms and
compare the direct vs inferential groups (and physical vs non-physical as a
secondary cut). No single metric is decisive on its own; convergence across
several is the point.

Metrics (per answer, aggregated over its content words):
  1. Zipf frequency        (wordfreq)          higher = more COMMON
  2. Concreteness          (Brysbaert 2014)    higher = more CONCRETE (less abstract)
  3. Word prevalence       (Brysbaert %known)  higher = fewer people find it NICHE
  4. Valence (neutrality)  (Warriner VAD)      abstract words are more affect-loaded
  5. Word length (chars)                       higher = more complex
  6. Syllable count        (CMUdict)           higher = more complex
  7. Answer token count                        higher = more verbose/specific
  8. Out-of-norms rate                         fraction of words absent from the
                                               40k-word norms = a direct RARITY proxy
"""
import json
import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from wordfreq import zipf_frequency
from nltk.corpus import cmudict
from nltk.stem import WordNetLemmatizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")

CMU = cmudict.dict()
LEM = WordNetLemmatizer()
STOP = {"a", "an", "the", "of", "and", "or", "with", "in", "on", "to", "for",
        "s", "its", "it", "is", "at", "by", "part"}

# ---- Norm tables -----------------------------------------------------------
conc = pd.read_csv(os.path.join(HERE, "norms/concreteness.csv"), sep="\t")
CONC = dict(zip(conc["Word"].astype(str).str.lower(), conc["Conc.M"]))
PKNOWN = dict(zip(conc["Word"].astype(str).str.lower(), conc["Percent_known"]))
vad = pd.read_csv(os.path.join(HERE, "norms/warriner_vad.csv"))
VAL = dict(zip(vad["Word"].astype(str).str.lower(), vad["V.Mean.Sum"]))


def syllables(w):
    w = w.lower()
    if w in CMU:
        return min(len([p for p in pr if p[-1].isdigit()]) for pr in CMU[w])
    # heuristic fallback: count vowel groups
    return max(1, len(re.findall(r"[aeiouy]+", w)))


def content_words(answer):
    """Primary alternative (before first '/'), stripped of parentheticals,
    lowercased, lemmatized, stopwords removed."""
    primary = answer.split("/")[0]
    primary = re.sub(r"\([^)]*\)", " ", primary)
    toks = re.findall(r"[a-zA-Z]+", primary.lower())
    words = [LEM.lemmatize(t) for t in toks if t not in STOP and len(t) > 1]
    return words or [LEM.lemmatize(t) for t in toks] or toks


def metrics_for_answer(answer):
    words = content_words(answer)
    if not words:
        return None
    zipf = [zipf_frequency(w, "en") for w in words]
    conc_v = [CONC[w] for w in words if w in CONC]
    pk_v = [PKNOWN[w] for w in words if w in PKNOWN]
    val_v = [VAL[w] for w in words if w in VAL]
    oov = sum(1 for w in words if w not in CONC) / len(words)
    return {
        "zipf": np.mean(zipf),
        "concreteness": np.mean(conc_v) if conc_v else np.nan,
        "pct_known": np.mean(pk_v) if pk_v else np.nan,
        "valence_neutrality": -abs(np.mean(val_v) - 5.0) if val_v else np.nan,  # closer to neutral = higher
        "word_len": np.mean([len(w) for w in words]),
        "syllables": np.mean([syllables(w) for w in words]),
        "n_tokens": len(words),
        "oov_rate": oov,
    }


# ---- Build the table -------------------------------------------------------
cat = pd.read_excel(os.path.join(ROOT, "data/Image categories (3).xlsx"))[["image id", "direct or inferential"]]
cat["image id"] = cat["image id"].astype(str)
di = dict(zip(cat["image id"], cat["direct or inferential"]))

ann = json.load(open(os.path.join(ROOT, "data/combined_output/annotations.json")))
rows = []
for e in ann:
    iid = str(e["id"])
    grp = di.get(iid)
    if grp not in ("direct", "inferential"):
        continue
    answer = (e.get("answer") or "").strip()
    if not answer:
        continue
    m = metrics_for_answer(answer)
    if m is None:
        continue
    m.update({"id": iid, "group": grp, "physical": e.get("physical", False),
              "answer": answer})
    rows.append(m)

df = pd.DataFrame(rows)
print(f"Answers scored: {len(df)}  "
      f"(direct={ (df.group=='direct').sum() }, inferential={ (df.group=='inferential').sum() })\n")

METRICS = ["zipf", "concreteness", "pct_known", "valence_neutrality",
           "word_len", "syllables", "n_tokens", "oov_rate"]
# For these, HIGHER means "more common / less niche / less abstract":
HIGHER_IS_COMMONER = {"zipf": True, "concreteness": True, "pct_known": True,
                      "valence_neutrality": True, "word_len": False,
                      "syllables": False, "n_tokens": False, "oov_rate": False}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return (gt - lt) / (len(a) * len(b))


LABEL = {"zipf": "Zipf frequency (common?)", "concreteness": "Concreteness (1-5)",
         "pct_known": "Word prevalence %known", "valence_neutrality": "Valence neutrality",
         "word_len": "Word length (chars)", "syllables": "Syllables",
         "n_tokens": "Answer token count", "oov_rate": "Out-of-norms rate"}


def bh_adjust(pvals):
    p = np.asarray(pvals); n = len(p); order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        k = n - rank
        prev = min(prev, p[idx] * n / k); adj[idx] = prev
    return adj


def compare(df, col_a, col_b, label):
    A = df[df.grpcol == col_a]; B = df[df.grpcol == col_b]
    print(f"===== {label}   ({col_a} n={len(A)}  vs  {col_b} n={len(B)}) =====")
    print(f"{'metric':<26}{'mean_'+col_a:>13}{'mean_'+col_b:>15}   MWU p    BH-p    Cliff d   verdict for "+col_a)
    raw = []
    for m in METRICS:
        a = A[m].dropna(); b = B[m].dropna()
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        raw.append(p)
    bh = bh_adjust(raw)
    for m, p, q in zip(METRICS, raw, bh):
        a = A[m].dropna(); b = B[m].dropna()
        d = cliffs_delta(a.values, b.values)          # +d => col_a values larger
        commoner = (d > 0) == HIGHER_IS_COMMONER[m]    # sign-based, tie-proof
        mag = abs(d)
        strength = "≈equal" if mag < 0.05 or q >= 0.05 else \
                   ("A more common/concrete" if commoner else "A more niche/abstract")
        star = "*" if q < 0.05 else " "
        print(f"{LABEL[m]:<26}{a.mean():>13.3f}{b.mean():>15.3f}  {p:>7.3g} {q:>6.3g}{star} {d:>+7.3f}   {strength}")
    print()


# Only comparison needed: direct/physical vs inferential/non-physical
# (in this dataset direct==physical and inferential==non-physical exactly).
df["grpcol"] = df["group"]
compare(df, "direct", "inferential",
        "DIRECT/PHYSICAL vs INFERENTIAL/NON-PHYSICAL  (reviewer's 'niche & abstract' claim)")

df.drop(columns=["grpcol"]).to_csv(os.path.join(HERE, "answer_lexical_metrics.csv"), index=False)
print("Per-answer metrics written to appendix/lexical_metrics/answer_lexical_metrics.csv")
print("\nNote: in this dataset every 'direct' answer is a physically-occluded image and\n"
      "every 'inferential' answer is digitally-occluded, so the two axes coincide.")
