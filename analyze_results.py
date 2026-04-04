import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

MODELS = ['llama-3.1-8b', 'llama-3.3-70b', 'deepseek-v3', 'gpt-oss-120b']
MODEL_LABELS = ['Llama 3.1\n(8B)', 'Llama 3.3\n(70B)', 'DeepSeek\nV3', 'GPT-OSS\n(120B)']
MODEL_LABELS_INLINE = ['Llama 3.1 (8B)', 'Llama 3.3 (70B)', 'DeepSeek V3', 'GPT-OSS (120B)']
PARAM_MAP = {'llama-3.1-8b': '8B', 'llama-3.3-70b': '70B', 'deepseek-v3': '685B (MoE)', 'gpt-oss-120b': '120B'}
COLORS = ['#D85A30', '#378ADD', '#7F77DD', '#1D9E75']
BENCH_COLORS = ['#5DCAA5', '#85B7EB']
FIGURE_DIR = Path('figures')
FIGURE_DIR.mkdir(exist_ok=True)

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def load_csv(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def find_csv():
    if len(sys.argv) > 1: return sys.argv[1]
    csvs = sorted(Path('results').glob('summary_*.csv'))
    if csvs: return str(csvs[-1])
    raise FileNotFoundError("No summary CSV found.")

def clean_data(rows):
    seen = set()
    clean = []
    for r in rows:
        key = (r['model'], r['task_id'])
        if key in seen: continue
        seen.add(key)
        if r['status'] in ('PASS', 'FAIL'): clean.append(r)
    return clean

def compute_pass_rates(rows):
    results = {}
    for m in MODELS:
        mr = [r for r in rows if r['model'] == m]
        if not mr: continue
        results[m] = {}
        for b, bl in zip(['humaneval', 'mbpp_sanitized'], ['HumanEval', 'MBPP']):
            s = [r for r in mr if r['benchmark'] == b]
            t = len(s); p = sum(1 for r in s if r['status'] == 'PASS')
            results[m][bl] = (p, t, round(100*p/t, 1) if t else 0)
        t = len(mr); p = sum(1 for r in mr if r['status'] == 'PASS')
        results[m]['Total'] = (p, t, round(100*p/t, 1) if t else 0)
    return results

def compute_errors(rows):
    fails = [r for r in rows if r['status'] == 'FAIL']
    return Counter(r['error_type'] for r in fails), {m: Counter(r['error_type'] for r in fails if r['model'] == m) for m in MODELS}, len(fails)

def fig1(pr):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    active = [m for m in MODELS if m in pr]
    x = np.arange(len(active)); w = 0.32
    he = [pr[m]['HumanEval'][2] for m in active]; mb = [pr[m]['MBPP'][2] for m in active]
    b1 = ax.bar(x-w/2, he, w, label='HumanEval', color=BENCH_COLORS[0], edgecolor='white')
    b2 = ax.bar(x+w/2, mb, w, label='MBPP', color=BENCH_COLORS[1], edgecolor='white')
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}%', xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,4), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
    ax.set_ylabel('pass@1 (%)'); ax.set_title('pass@1 by Model and Benchmark')
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[MODELS.index(m)] for m in active])
    ax.set_ylim(0, 105); ax.legend(loc='upper left'); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(FIGURE_DIR / 'fig1_pass_by_benchmark.pdf'); fig.savefig(FIGURE_DIR / 'fig1_pass_by_benchmark.png'); plt.close(fig)
    print("  fig1_pass_by_benchmark")

def fig2(pr):
    fig, ax = plt.subplots(figsize=(7, 4))
    active = [m for m in MODELS if m in pr]
    vals = [pr[m]['Total'][2] for m in active]
    bars = ax.bar([MODEL_LABELS[MODELS.index(m)] for m in active], vals, color=[COLORS[MODELS.index(m)] for m in active], edgecolor='white', width=0.55)
    for bar, v in zip(bars, vals):
        ax.annotate(f'{v:.1f}%', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0,5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('pass@1 (%)'); ax.set_title('Overall pass@1 by Model'); ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(FIGURE_DIR / 'fig2_pass_overall.pdf'); fig.savefig(FIGURE_DIR / 'fig2_pass_overall.png'); plt.close(fig)
    print("  fig2_pass_overall")

def fig3(overall, total_fails):
    fig, ax = plt.subplots(figsize=(7, 4))
    se = sorted(overall.items(), key=lambda x: x[1])
    labels = [e[0] for e in se]; counts = [e[1] for e in se]; pcts = [100*c/total_fails for c in counts]
    colors_bar = ['#378ADD' if l != 'AssertionError' else '#D85A30' for l in labels]
    bars = ax.barh(labels, counts, color=colors_bar, edgecolor='white', height=0.6)
    for bar, pct in zip(bars, pcts):
        ax.annotate(f'{pct:.1f}%', xy=(bar.get_width(), bar.get_y()+bar.get_height()/2), xytext=(4,0), textcoords="offset points", ha='left', va='center', fontsize=9)
    ax.set_xlabel('Number of Failures'); ax.set_title(f'Error Type Distribution (n={total_fails} failures)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(axis='x', alpha=0.3)
    fig.savefig(FIGURE_DIR / 'fig3_error_distribution.pdf'); fig.savefig(FIGURE_DIR / 'fig3_error_distribution.png'); plt.close(fig)
    print("  fig3_error_distribution")

def fig4(per_model):
    all_e = Counter()
    for m in MODELS: all_e.update(per_model.get(m, {}))
    top = [e for e, c in all_e.most_common() if c >= 3]
    active = [m for m in MODELS if sum(per_model.get(m, {}).values()) > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(top)); w = 0.2
    for i, m in enumerate(active):
        vals = [per_model[m].get(e, 0) for e in top]
        ax.bar(x+(i-len(active)/2+0.5)*w, vals, w, label=MODEL_LABELS_INLINE[MODELS.index(m)], color=COLORS[MODELS.index(m)], edgecolor='white')
    ax.set_ylabel('Number of Failures'); ax.set_title('Error Types by Model')
    ax.set_xticks(x); ax.set_xticklabels(top, rotation=30, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=8); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(FIGURE_DIR / 'fig4_errors_by_model.pdf'); fig.savefig(FIGURE_DIR / 'fig4_errors_by_model.png'); plt.close(fig)
    print("  fig4_errors_by_model")

def fig5(pr):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sizes = {'llama-3.1-8b': 8, 'llama-3.3-70b': 70, 'deepseek-v3': 685, 'gpt-oss-120b': 120}
    active = [m for m in MODELS if m in pr]
    pts = sorted([(sizes[m], pr[m]['Total'][2], m) for m in active])
    for s, t, m in pts:
        idx = MODELS.index(m)
        ax.scatter(s, t, s=120, color=COLORS[idx], zorder=5, edgecolors='white', linewidth=1.5)
        oy = 12 if t < 93 else -18
        ax.annotate(f'{t:.1f}%\n{MODEL_LABELS_INLINE[idx]}', xy=(s, t), xytext=(0, oy), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('Model Size (Billion Parameters)'); ax.set_ylabel('pass@1 (%)')
    ax.set_title('Model Size vs. Code Generation Accuracy'); ax.set_ylim(50, 100); ax.set_xlim(-20, 750)
    ax.grid(alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(FIGURE_DIR / 'fig5_size_vs_pass.pdf'); fig.savefig(FIGURE_DIR / 'fig5_size_vs_pass.png'); plt.close(fig)
    print("  fig5_size_vs_pass")

def run_stats(rows, per_model, overall):
    print("\n=== STATISTICAL TESTS ===")
    active = [m for m in MODELS if any(r['model'] == m for r in rows)]
    table = []
    for m in active:
        mr = [r for r in rows if r['model'] == m]
        p = sum(1 for r in mr if r['status'] == 'PASS')
        table.append([p, len(mr) - p])
    chi2, pv, dof, _ = stats.chi2_contingency(np.array(table))
    print(f"  Chi-square (pass/fail): chi2={chi2:.2f}, p={pv:.2e}, dof={dof}")
    top_e = [e for e, c in overall.most_common() if c >= 3]
    cont = [[per_model.get(m, {}).get(e, 0) for e in top_e] for m in active]
    chi2_e, pe, de, _ = stats.chi2_contingency(np.array(cont))
    print(f"  Chi-square (error types): chi2={chi2_e:.2f}, p={pe:.4f}, dof={de}")

def main():
    csv_path = find_csv()
    print(f"Loading: {csv_path}")
    rows = clean_data(load_csv(csv_path))
    print(f"Clean records: {len(rows)}")
    pr = compute_pass_rates(rows)
    overall, per_model, total_fails = compute_errors(rows)
    print("\nSummary:")
    for m in MODELS:
        if m in pr:
            p = pr[m]; print(f"  {MODEL_LABELS_INLINE[MODELS.index(m)]}: HE={p['HumanEval'][2]}%, MBPP={p['MBPP'][2]}%, Total={p['Total'][2]}%")
    print(f"  Failures: {total_fails}")
    print("\nGenerating figures...")
    fig1(pr); fig2(pr); fig3(overall, total_fails); fig4(per_model); fig5(pr)
    run_stats(rows, per_model, overall)
    print(f"\nDone! Figures in {FIGURE_DIR}/")

if __name__ == "__main__":
    main()
