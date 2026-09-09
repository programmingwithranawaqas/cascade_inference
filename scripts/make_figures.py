"""Regenerate every figure in the paper.

    python scripts/make_figures.py --probs probs --costs results/costs.json
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import linregress

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlfnd.cascade import product_ensemble
from mlfnd.data import LANGUAGES
from mlfnd.metrics import accuracy, difficulty_inversion_score

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linewidth': 0.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.dpi': 300, 'savefig.bbox': 'tight',
})
PALETTE = {'gate': '#1b4965', 'bilstm': '#5fa8d3', 'xlmr': '#c1666b', 'muril': '#e8a87c'}


def save(fig, out_dir, name):
    for extension in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{extension}'))
    plt.close(fig)
    print(f"  wrote {name}")


def figure_cascade(out_dir, costs):
    cost_of = lambda m: costs['models'][m]['fixed']['by_batch']['32']['end_to_end']['median_ms_per_instance']
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
    tiers = [(0.4, 'Tier 1\nTF-IDF gate', cost_of('gate'), PALETTE['gate']),
             (3.7, 'Tier 2\nXLM-R', cost_of('xlmr'), PALETTE['xlmr']),
             (7.0, 'Tier 3\nensemble', cost_of('xlmr') + cost_of('muril'), PALETTE['muril'])]
    for x, label, cost, colour in tiers:
        ax.add_patch(FancyBboxPatch((x, 1.0), 2.3, 1.2, boxstyle="round,pad=0.06",
                                    fc=colour, ec='none', alpha=0.85))
        ax.text(x + 1.15, 1.78, label, ha='center', va='center', color='w',
                fontsize=8.5, weight='bold')
        ax.text(x + 1.15, 1.25, f'{cost:.4f} ms', ha='center', va='center',
                color='w', fontsize=7.5)
    for x0, x1, label in [(2.7, 3.7, '11.4%'), (6.0, 7.0, '0.5%')]:
        ax.add_patch(FancyArrowPatch((x0, 1.6), (x1, 1.6), arrowstyle='-|>',
                                     mutation_scale=11, lw=1.2, color='#333'))
        ax.text((x0 + x1) / 2, 1.80, label, ha='center', fontsize=7.5, color='#333')
    for x, label in [(1.55, '88.6% exit'), (4.85, '11.0% exit')]:
        ax.add_patch(FancyArrowPatch((x, 1.0), (x, 0.42), arrowstyle='-|>',
                                     mutation_scale=11, lw=1.2, color='#777'))
        ax.text(x + 0.12, 0.60, label, ha='left', fontsize=7.5, color='#555')
    ax.text(0.4, 2.55, r'escalate if  $\max_y\, p(y \mid x) \leq \tau_m$',
            fontsize=8.5, color='#333')
    save(fig, out_dir, 'fig1_cascade')


def figure_costs(out_dir, costs):
    batch_sizes = costs['batch_sizes']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, padding, title in zip(axes, ['fixed', 'dynamic'],
                                  ['Fixed padding', 'Dynamic padding']):
        for key, label in [('gate', 'TF-IDF gate'), ('bilstm', 'Bi-LSTM'),
                           ('xlmr', 'XLM-R'), ('muril', 'MuRIL')]:
            block = costs['models'][key].get(padding) or costs['models'][key].get('fixed')
            entries = [block['by_batch'][str(b)]['end_to_end'] for b in batch_sizes]
            median = [e['median_ms_per_instance'] for e in entries]
            ax.plot(batch_sizes, median, 'o-', ms=3.5, lw=1.3,
                    color=PALETTE[key], label=label)
            ax.fill_between(batch_sizes,
                            [e['p25_ms_per_instance'] for e in entries],
                            [e['p75_ms_per_instance'] for e in entries],
                            color=PALETTE[key], alpha=0.18, lw=0)
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xticks(batch_sizes); ax.set_xticklabels(batch_sizes)
        ax.set_xlabel('batch size'); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel('ms per instance (end-to-end)')
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper right')
    save(fig, out_dir, 'fig2_cost_sweep')


def figure_frontier(out_dir, rows):
    """rows: {scheme: (accuracies, speedups)}"""
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    styles = {'shared threshold': ('#5fa8d3', 'o'),
              'shared + per-model scaling': ('#c1666b', 's'),
              'per-tier thresholds': ('#1b4965', '^')}
    for label, (accs, speedups) in rows.items():
        colour, marker = styles.get(label, ('#888', 'o'))
        ax.plot(speedups, accs, marker + '-', ms=4, lw=1.3, color=colour, label=label)
    ax.set_xscale('log')
    ax.set_xlabel(r'speedup over the ensemble ($\times$)')
    ax.set_ylabel('test accuracy (%)')
    ax.legend(frameon=False, fontsize=7.5, loc='lower left')
    save(fig, out_dir, 'fig3_frontier')


def figure_seed_spread(out_dir, probs_dir, y_test):
    groups = [('recurrent\ncross-entropy',
               sorted(glob.glob(os.path.join(probs_dir, 'probs_bilstm_ce_seed*.npz'))), '#4a4a4a')]
    for epsilon, colour in [(0.1, '#5fa8d3'), (0.5, '#1b4965')]:
        paths = sorted(glob.glob(os.path.join(
            probs_dir, f'probs_bilstm_dar_eps{epsilon}_seed*.npz')))
        if paths:
            groups.append((f'+ margin\n$\\varepsilon$={epsilon}', paths, colour))
    for pattern, label, colour in [
            ('probs_muril_baseline*.npz', 'MuRIL\ncross-entropy', '#4a4a4a'),
            ('probs_muril_logitnorm*.npz', 'MuRIL\n+ logit norm', '#c1666b')]:
        paths = sorted(glob.glob(os.path.join(probs_dir, pattern)))
        if paths:
            groups.append((label, paths, colour))
    groups = [(label, paths, colour) for label, paths, colour in groups if paths]
    if not groups:
        print("  no seed arms present, skipping the spread figure")
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    rng = np.random.default_rng(0)
    for i, (label, paths, colour) in enumerate(groups):
        values = [difficulty_inversion_score(
            np.load(p, allow_pickle=True)['probs_test'], y_test) for p in paths]
        ax.scatter(np.full(len(values), i) + rng.uniform(-0.09, 0.09, len(values)),
                   values, s=26, color=colour, zorder=3, edgecolor='w', linewidth=0.5)
        ax.hlines(np.mean(values), i - 0.26, i + 0.26, color=colour, lw=2, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel('difficulty inversion score')
    save(fig, out_dir, 'fig4_seed_spread')


def figure_mechanism(out_dir, gate_test, y_test, lang_test, hb_gate, hb_y):
    offsets = {'Code_Mixed': (6, 7), 'English': (6, -12),
               'Roman_Urdu': (-8, -14), 'Urdu': (8, 4)}
    accs, rates = [], []
    fig, ax = plt.subplots(figsize=(4.9, 3.5))
    for language in LANGUAGES:
        mask = lang_test == language
        accs.append(accuracy(gate_test[mask], y_test[mask]))
        rates.append((gate_test[mask].max(1) <= 0.845).mean() * 100)
    fit = linregress(accs, rates)
    xs = np.linspace(min(accs) - 6, max(accs) + 1, 50)
    ax.plot(xs, fit.slope * xs + fit.intercept, '-', lw=1.2, color='#999',
            label=f'fit on language strata ($r$ = {fit.rvalue:.4f})')
    ax.scatter(accs, rates, s=46, color='#1b4965', zorder=3, label='language strata')
    for language, x, y in zip(LANGUAGES, accs, rates):
        ax.annotate(language.replace('_', ' '), (x, y), textcoords='offset points',
                    xytext=offsets[language], fontsize=7.5, color='#1b4965')
    if hb_gate is not None:
        held_acc = accuracy(hb_gate, hb_y)
        held_rate = (hb_gate.max(1) <= 0.845).mean() * 100
        predicted = fit.slope * held_acc + fit.intercept
        ax.scatter([held_acc], [held_rate], s=90, marker='D', color='#c1666b',
                   zorder=5, edgecolor='w', linewidth=0.8, label='replication (held out)')
        ax.scatter([held_acc], [predicted], s=52, marker='x', color='#c1666b',
                   zorder=5, lw=1.6)
        ax.annotate('', xy=(held_acc, held_rate - 0.35), xytext=(held_acc, predicted + 0.35),
                    arrowprops=dict(arrowstyle='<->', color='#c1666b', lw=1))
        ax.text(held_acc + 0.35, (held_rate + predicted) / 2,
                f'residual {abs(held_rate - predicted):.2f} pp',
                fontsize=7.5, color='#c1666b', va='center')
    ax.set_xlabel('tier-1 accuracy (%)')
    ax.set_ylabel(r'escalation rate at $\tau$ = 0.845 (%)')
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    save(fig, out_dir, 'fig5_mechanism')


def frontier_rows(bundle, gate, y_val, y_test, costs,
                  tolerances=(0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.00)):
    """Recompute the frontier of Table 7 so the figure cannot drift from it."""
    from mlfnd.cascade import COARSE_GRID, cost_three_tier, route_three_tier
    from mlfnd.metrics import accuracy as acc_of, fit_temperature, temperature_scale
    cost_of = lambda m: costs['models'][m]['fixed']['by_batch']['32']['end_to_end']['median_ms_per_instance']
    c_gate, c_mid, c_top = cost_of('gate'), cost_of('xlmr'), cost_of('muril')
    c_full = c_mid + c_top
    ens_val = product_ensemble(bundle['pa_val'], bundle['pb_val'])
    ens_test = product_ensemble(bundle['pa_test'], bundle['pb_test'])
    temps = {'gate': fit_temperature(gate['probs_val'], y_val),
             'mid': fit_temperature(bundle['pb_val'], y_val),
             'top': fit_temperature(bundle['pa_val'], y_val)}
    schemes = {
        'shared threshold': (gate['probs_val'], gate['probs_test'], bundle['pb_val'],
                             bundle['pb_test'], ens_val, ens_test, True),
        'shared + per-model scaling': (
            temperature_scale(gate['probs_val'], temps['gate']),
            temperature_scale(gate['probs_test'], temps['gate']),
            temperature_scale(bundle['pb_val'], temps['mid']),
            temperature_scale(bundle['pb_test'], temps['mid']),
            temperature_scale(ens_val, temps['top']),
            temperature_scale(ens_test, temps['top']), True),
        'per-tier thresholds': (gate['probs_val'], gate['probs_test'], bundle['pb_val'],
                                bundle['pb_test'], ens_val, ens_test, False),
    }
    rows = {}
    for label, (gv, gt, mv, mt, ev, et, shared) in schemes.items():
        accs, speedups = [], []
        candidates = ([(t, t) for t in COARSE_GRID] if shared
                      else [(a, b) for a in COARSE_GRID for b in COARSE_GRID])
        for tolerance in tolerances:
            best = None
            for tau_gate, tau_mid in candidates:
                predictions, exits = route_three_tier(gv, mv, ev, tau_gate, tau_mid)
                if (predictions == y_val).mean() * 100 >= acc_of(ev, y_val) - tolerance:
                    cost = cost_three_tier(exits, c_gate, c_mid, c_top)
                    if best is None or cost < best[1]:
                        best = ((tau_gate, tau_mid), cost)
            predictions, exits = route_three_tier(gt, mt, et, *best[0])
            accs.append((predictions == y_test).mean() * 100)
            speedups.append(c_full / cost_three_tier(exits, c_gate, c_mid, c_top))
        rows[label] = (accs, speedups)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probs', default='probs')
    parser.add_argument('--probs_hb', default='probs_hb')
    parser.add_argument('--costs', default='results/costs.json')
    parser.add_argument('--out_dir', default='figures')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    costs = json.load(open(args.costs))
    bundle = np.load(os.path.join(args.probs, 'probs_bundle.npz'), allow_pickle=True)
    gate = np.load(os.path.join(args.probs, 'probs_tfidf.npz'), allow_pickle=True)
    y_test = bundle['y_test']
    lang_test = bundle['lang_test'].astype(str)

    hb_path = os.path.join(args.probs_hb, 'probs_hb_tfidf.npz')
    hb_bundle_path = os.path.join(args.probs_hb, 'probs_hb_bundle.npz')
    hb_gate = hb_y = None
    if os.path.exists(hb_path) and os.path.exists(hb_bundle_path):
        hb_gate = np.load(hb_path, allow_pickle=True)['probs_test']
        hb_y = np.load(hb_bundle_path, allow_pickle=True)['y_test']

    figure_cascade(args.out_dir, costs)
    figure_costs(args.out_dir, costs)
    figure_frontier(args.out_dir, frontier_rows(bundle, gate, y_val=bundle['y_val'],
                                                y_test=y_test, costs=costs))
    figure_seed_spread(args.out_dir, args.probs, y_test)
    figure_mechanism(args.out_dir, gate['probs_test'], y_test, lang_test, hb_gate, hb_y)


if __name__ == '__main__':
    main()
