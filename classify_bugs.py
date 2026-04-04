import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

random.seed(42)

FIGURE_DIR = Path('figures')
FIGURE_DIR.mkdir(exist_ok=True)

MODELS = ['llama-3.1-8b', 'llama-3.3-70b', 'deepseek-v3', 'gpt-oss-120b']
MODEL_LABELS = ['Llama 3.1\n(8B)', 'Llama 3.3\n(70B)', 'DeepSeek\nV3', 'GPT-OSS\n(120B)']

TAMBON = {
    'Misinterpretation': 20.8, 'Missing Corner Case': 15.3, 'Silly Mistake': 9.6,
    'Hallucinated Object': 9.6, 'Incomplete Generation': 9.6, 'Wrong Attribute': 8.6,
    'Non-Prompted Consideration': 8.2, 'Prompt-biased Code': 6.5,
    'Syntax Error': 6.1, 'Wrong Input Type': 5.9,
}

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def find_tested():
    if len(sys.argv) > 1: return sys.argv[1]
    fs = sorted(Path('results').glob('tested_results_*.json'))
    if fs: return str(fs[-1])
    raise FileNotFoundError("No tested_results JSON found.")


def classify(prompt, code, error_type, error_msg):
    code = code or ''; prompt = prompt or ''; error_msg = error_msg or ''
    cl, pl = code.lower(), prompt.lower()
    lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]

    if error_type == 'SyntaxError': return 'Syntax Error'
    if error_type == 'NameError': return 'Hallucinated Object'
    if error_type == 'AttributeError': return 'Wrong Attribute'
    if error_type in ('TypeError', 'ValueError'): return 'Wrong Input Type'
    if error_type == 'TimeoutError': return 'Silly Mistake'
    if error_type == 'ZeroDivisionError': return 'Missing Corner Case'
    if error_type == 'IndexError': return 'Missing Corner Case'

    if error_type == 'AssertionError':
        if not code.strip() or code.strip() == 'pass' or len(lines) <= 1:
            return 'Incomplete Generation'
        for op in ['sorted(', '.sort()', '.reverse()', 'abs(', 'round(', '.lower()', '.upper()', '.strip()']:
            if op in cl and op not in pl:
                return 'Non-Prompted Consideration'
        if any(x in code for x in ['return [', 'return {', 'return (']) and len(lines) <= 5:
            return 'Prompt-biased Code'
        fp = fc = ''
        for l in prompt.split('\n'):
            if 'def ' in l: fp = l.split('def ')[1].split('(')[0].strip(); break
        for l in code.split('\n'):
            if 'def ' in l: fc = l.split('def ')[1].split('(')[0].strip(); break
        if fp and fc and fp != fc: return 'Misinterpretation'
        if len(lines) >= 5: return 'Missing Corner Case'
        return 'Misinterpretation'

    return 'Misinterpretation'


def main():
    path = find_tested()
    print(f"Loading: {path}")
    with open(path) as f: data = json.load(f)

    # Remove duplicates
    seen = set()
    unique = []
    for r in data:
        key = (r['model'], r['task_id'])
        if key in seen: continue
        seen.add(key)
        unique.append(r)

    fails = [r for r in unique if r['test_result']['status'] == 'FAIL']
    print(f"Total failures: {len(fails)}")

    # Stratified sample
    sample = []
    for model, n in [('llama-3.1-8b', 25), ('llama-3.3-70b', 20), ('deepseek-v3', 15), ('gpt-oss-120b', 25)]:
        mf = [r for r in fails if r['model'] == model]
        sample.extend(mf[:n] if len(mf) <= n else random.sample(mf, n))

    print(f"Sample: {len(sample)}")

    classified = []
    for r in sample:
        cat = classify(r.get('prompt', ''), r.get('generated_code', ''),
                       r['test_result']['error_type'], r['test_result'].get('error_message', ''))
        classified.append({'task_id': r['task_id'], 'model': r['model'],
                          'error_type': r['test_result']['error_type'], 'tambon_category': cat})

    total = len(classified)
    cats = Counter(r['tambon_category'] for r in classified)
    per_model = {}
    model_totals = {}
    for m in MODELS:
        mc = [r for r in classified if r['model'] == m]
        per_model[m] = Counter(r['tambon_category'] for r in mc)
        model_totals[m] = len(mc)

    print(f"\n{'Category':<32} {'Count':>5} {'%':>7}")
    print("-" * 46)
    for c, n in cats.most_common():
        print(f"{c:<32} {n:>5} {100*n/total:>6.1f}%")

    print(f"\nComparison with Tambon:")
    print(f"{'Category':<32} {'Tambon':>8} {'Ours':>8}")
    print("-" * 50)
    for c in TAMBON:
        print(f"{c:<32} {TAMBON[c]:>7.1f}% {100*cats.get(c,0)/total:>7.1f}%")

    # Fig 7: Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    all_cats = list(TAMBON.keys())
    tp = [TAMBON[c] for c in all_cats]; op = [100*cats.get(c,0)/total for c in all_cats]
    x = np.arange(len(all_cats)); w = 0.35
    ax.bar(x-w/2, tp, w, label='Tambon et al. (2025)', color='#85B7EB', edgecolor='white')
    ax.bar(x+w/2, op, w, label='Our Study', color='#D85A30', edgecolor='white')
    ax.set_ylabel('Percentage (%)'); ax.set_title('Bug Pattern Distribution: Tambon et al. vs. Our Study')
    ax.set_xticks(x); ax.set_xticklabels([c.replace(' ', '\n') if len(c) > 12 else c for c in all_cats], fontsize=8)
    ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(FIGURE_DIR / 'fig7_tambon_comparison.pdf'); fig.savefig(FIGURE_DIR / 'fig7_tambon_comparison.png'); plt.close(fig)
    print("\n  Saved fig7_tambon_comparison")

    # Fig 8: Per model
    fig, ax = plt.subplots(figsize=(8, 5))
    top5 = ['Missing Corner Case', 'Misinterpretation', 'Non-Prompted Consideration', 'Prompt-biased Code', 'Silly Mistake', 'Other']
    cat_colors = ['#D85A30', '#1D9E75', '#7F77DD', '#EF9F27', '#D4537E', '#73726c']
    active = [m for m in MODELS if model_totals.get(m, 0) > 0]
    x = np.arange(len(active)); bottom = np.zeros(len(active))
    for cat, color in zip(top5, cat_colors):
        if cat == 'Other':
            vals = [100*sum(v for k, v in per_model[m].items() if k not in top5[:5])/model_totals[m] if model_totals[m] else 0 for m in active]
        else:
            vals = [100*per_model[m].get(cat, 0)/model_totals[m] if model_totals[m] else 0 for m in active]
        ax.bar(x, vals, 0.5, bottom=bottom, label=cat, color=color, edgecolor='white')
        bottom += np.array(vals)
    ax.set_ylabel('Percentage (%)'); ax.set_title('Bug Patterns by Model (Tambon Taxonomy)')
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[MODELS.index(m)] for m in active])
    ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.35, 1))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(FIGURE_DIR / 'fig8_tambon_per_model.pdf'); fig.savefig(FIGURE_DIR / 'fig8_tambon_per_model.png'); plt.close(fig)
    print("  Saved fig8_tambon_per_model")

    with open('results/bug_classification.json', 'w') as f:
        json.dump({'sample_size': total, 'overall': dict(cats), 'per_model': {m: dict(c) for m, c in per_model.items()}, 'classified': classified}, f, indent=2)
    print("  Saved results/bug_classification.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
