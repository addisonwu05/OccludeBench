"""
Direct+intervention vs inferential-baseline gap, final convention:
  * Extended reasoning / tool use / few-shot : intervention applied to ALL direct images (measured)
  * Inpaint-then-confront                    : recovery arm — baseline-right kept, baseline-wrong take the confront verdict
Rates binarised per image (mean of gpt-5.4 + claude-opus-4-6 judge verdicts >= 0.5). Two-sided Fisher exact test.
"""
import json, os
from scipy.stats import fisher_exact
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JUDGES = ['gpt-5.4', 'claude-opus-4-6']
NAMES = {'gpt-5.4': 'GPT-5.4', 'claude-opus-4-6': 'Claude Opus 4.6', 'gemini-3.1-pro-preview': 'Gemini 3.1 Pro',
         'Qwen3-VL-8B-Instruct': 'Qwen3-VL 8B', 'Llama-4-Maverick-17B-128E-Instruct-FP8-9940db5c': 'Llama-4 Maverick'}
ALL = list(NAMES)
INTERVENTIONS = {
    'Extended reasoning':    ('data/eval_results_thinking',          ['gpt-5.4', 'claude-opus-4-6'], 'all'),
    'Tool use (+thinking)':  ('data/eval_results_tool_use_thinking', ['gpt-5.4', 'claude-opus-4-6'], 'all'),
    'Few-shot':              ('data/eval_results_fewshot',           ALL,                            'all'),
    'Inpaint-then-confront': ('sec4_3_improvements/inpainting_multistep', ['gpt-5.4', 'gemini-3.1-pro-preview'], 'recovery'),
}
physical = {str(a['id']): bool(a['physical']) for a in json.load(open(os.path.join(ROOT, 'data/combined_output/annotations.json')))}
def binmap(dir_, stem):
    out = {}
    for it in json.load(open(os.path.join(ROOT, dir_, f'{stem}_judged_results.json'))):
        v = it.get('occluded_correct') or {}
        s = [v[j] for j in JUDGES if j in v and v[j] is not None]
        if s: out[str(it['id'])] = int(sum(s) / len(s) >= 0.5)
    return out
base = {m: binmap('data/eval_results', m) for m in ALL}
summary = {}
for name, (dir_, models, mode) in INTERVENTIONS.items():
    print(f"=== {name}  ({'all direct images' if mode=='all' else 'baseline-wrong only, baseline-right kept'}) ===")
    print(f"{'Model':18s} {'dir base':>9s} {'dir+int':>12s} {'inf base':>12s} {'gap':>7s} {'Fisher p':>9s}  sig?")
    n_ns = 0
    for m in models:
        b, f = base[m], binmap(dir_, m)
        d = [i for i in b if physical[i]]; inf = [i for i in b if not physical[i]]
        if mode == 'all':
            fd = [i for i in f if physical.get(i)]; k, n = sum(f[i] for i in fd), len(fd)
        else:
            right = [i for i in d if b[i]]; wrong = [i for i in d if not b[i] and i in f]
            k, n = len(right) + sum(f[i] for i in wrong), len(right) + len(wrong)
        ki, ni = sum(b[i] for i in inf), len(inf)
        p = fisher_exact([[k, n - k], [ki, ni - ki]])[1]; ns = p >= 0.05; n_ns += ns
        print(f"{NAMES[m]:18s} {100*sum(b[i] for i in d)/len(d):8.1f}% {k:>4d}/{n:<3d}={100*k/n:5.1f}% {ki:>4d}/{ni:<3d}={100*ki/ni:5.1f}% {100*(k/n-ki/ni):+6.1f} {p:9.4f}  {'NOT sig' if ns else 'sig'}")
    summary[name] = (n_ns, len(models)); print(f"-> {n_ns}/{len(models)} non-significant\n")
print("==== SUMMARY ====")
for k, (n, t) in summary.items(): print(f"{k:24s} {n}/{t}")
