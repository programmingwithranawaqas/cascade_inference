"""Regenerate every table in the paper from the saved arrays and the cost file.

One script, one pass, so that the manuscript and the released code cannot drift
apart. Run it and copy the output.

    python scripts/make_tables.py --probs probs --costs results/costs.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import linregress, ttest_ind

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlfnd.cascade import (apply_per_language, cost_three_tier, cost_two_tier,
                           product_ensemble, route_three_tier, route_two_tier,
                           select_tau_per_language, select_tau_three_tier,
                           select_tau_two_tier, COARSE_GRID, DEFAULT_GRID)
from mlfnd.certificates import calibrate, damage_rate, smallest_certifiable_alpha
from mlfnd.data import LANGUAGES
from mlfnd.metrics import (accuracy, difficulty_inversion_score,
                           escalation_sets_identical, expected_calibration_error,
                           fit_temperature, mcnemar, temperature_scale)

BATCH = 32


def cost_of(costs, model, batch=BATCH, regime='end_to_end', padding='fixed'):
    return costs['models'][model][padding]['by_batch'][str(batch)][regime]['median_ms_per_instance']


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probs', default='probs')
    parser.add_argument('--probs_hb', default='probs_hb')
    parser.add_argument('--costs', default='results/costs.json')
    args = parser.parse_args()

    costs = json.load(open(args.costs))
    c_gate = cost_of(costs, 'gate')
    c_mid = cost_of(costs, 'xlmr')
    c_top = cost_of(costs, 'muril')
    c_full = c_mid + c_top

    bundle = np.load(os.path.join(args.probs, 'probs_bundle.npz'), allow_pickle=True)
    gate = np.load(os.path.join(args.probs, 'probs_tfidf.npz'), allow_pickle=True)
    lstm = np.load(os.path.join(args.probs, 'probs_bilstm.npz'), allow_pickle=True)
    y_val, y_test = bundle['y_val'], bundle['y_test']
    lang_val = bundle['lang_val'].astype(str)
    lang_test = bundle['lang_test'].astype(str)
    gate_val, gate_test = gate['probs_val'], gate['probs_test']
    mid_val, mid_test = bundle['pb_val'], bundle['pb_test']
    ens_val = product_ensemble(bundle['pa_val'], bundle['pb_val'])
    ens_test = product_ensemble(bundle['pa_test'], bundle['pb_test'])
    ref_val, ref_test = accuracy(ens_val, y_val), accuracy(ens_test, y_test)

    rule("Table 2  model inventory")
    print(f"{'model':<28}{'test acc':>10}{'ECE':>9}{'DIS':>8}")
    for name, probs in [('tier-1 gate', gate_test), ('recurrent tier-1', lstm['probs_test']),
                        ('XLM-R', bundle['pb_test']), ('MuRIL', bundle['pa_test']),
                        ('two-way ensemble', ens_test)]:
        print(f"{name:<28}{accuracy(probs, y_test):>10.2f}"
              f"{expected_calibration_error(probs, y_test):>9.4f}"
              f"{difficulty_inversion_score(probs, y_test):>8.2f}")

    rule("Table 4  per-instance cost, end-to-end, fixed padding")
    header = f"{'batch':>6}" + ''.join(f"{m:>11}" for m in ['gate', 'bilstm', 'xlmr', 'muril'])
    print(header)
    for batch in costs['batch_sizes']:
        row = f"{batch:>6}"
        for model in ['gate', 'bilstm', 'xlmr', 'muril']:
            row += f"{cost_of(costs, model, batch):>11.4f}"
        print(row)

    rule("Table 5  cascade configurations")
    tau = 0.845
    predictions, escalated = route_two_tier(gate_test, ens_test, tau)
    cost = cost_two_tier(escalated, c_gate, c_full)
    a, b, p = mcnemar(ens_test.argmax(1), predictions, y_test)
    print(f"two-tier  tau={tau}  escalation {escalated.mean() * 100:.2f}%  "
          f"acc {(predictions == y_test).mean() * 100:.2f}  "
          f"speedup {c_full / cost:.2f}x  McNemar b01={a} b10={b} p={p:.3f}")
    for tolerance in [0.10, 0.25]:
        chosen = select_tau_three_tier(gate_val, mid_val, ens_val, y_val, ref_val,
                                       tolerance, (c_gate, c_mid, c_top))
        predictions, exits = route_three_tier(gate_test, mid_test, ens_test, *chosen)
        cost = cost_three_tier(exits, c_gate, c_mid, c_top)
        print(f"three-tier  eps={tolerance}pp  tau={chosen}  "
              f"acc {(predictions == y_test).mean() * 100:.2f}  "
              f"speedup {c_full / cost:.2f}x  "
              f"exits {exits[0].mean() * 100:.1f}/{exits[1].mean() * 100:.1f}/"
              f"{exits[2].mean() * 100:.1f}%")

    rule("Table 6  tier-1 comparison at matched accuracy")
    for tolerance in [0.10, 0.25]:
        for name, val_probs, test_probs, unit_cost in [
                ('gate', gate_val, gate_test, c_gate),
                ('recurrent', lstm['probs_val'], lstm['probs_test'], cost_of(costs, 'bilstm'))]:
            tau = select_tau_two_tier(val_probs, ens_val, y_val, ref_val,
                                      tolerance, unit_cost, c_full)
            if tau is None:
                print(f"  eps={tolerance}pp  {name:<10} infeasible")
                continue
            predictions, escalated = route_two_tier(test_probs, ens_test, tau)
            cost = cost_two_tier(escalated, unit_cost, c_full)
            print(f"  eps={tolerance}pp  {name:<10} tau={tau:.4f}  "
                  f"escalation {escalated.mean() * 100:5.2f}%  "
                  f"acc {(predictions == y_test).mean() * 100:.2f}  "
                  f"speedup {c_full / cost:5.2f}x")

    rule("Table 7  frontier under three threshold schemes")
    temps = {'gate': fit_temperature(gate_val, y_val),
             'mid': fit_temperature(mid_val, y_val),
             'top': fit_temperature(bundle['pa_val'], y_val)}
    print(f"fitted temperatures: {temps}")
    schemes = {
        'shared threshold': (gate_val, gate_test, mid_val, mid_test, ens_val, ens_test, True),
        'shared + per-model scaling': (
            temperature_scale(gate_val, temps['gate']), temperature_scale(gate_test, temps['gate']),
            temperature_scale(mid_val, temps['mid']), temperature_scale(mid_test, temps['mid']),
            temperature_scale(ens_val, temps['top']), temperature_scale(ens_test, temps['top']), True),
        'per-tier thresholds': (gate_val, gate_test, mid_val, mid_test, ens_val, ens_test, False),
    }
    print(f"{'eps':>6}" + ''.join(f"{k:>30}" for k in schemes))
    for tolerance in [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.00]:
        row = f"{tolerance:>6.2f}"
        for gv, gt, mv, mt, ev, et, shared in schemes.values():
            grid = [(t, t) for t in COARSE_GRID] if shared else None
            best = None
            candidates = grid or [(a, b) for a in COARSE_GRID for b in COARSE_GRID]
            for tau_gate, tau_mid in candidates:
                predictions, exits = route_three_tier(gv, mv, ev, tau_gate, tau_mid)
                if (predictions == y_val).mean() * 100 >= accuracy(ev, y_val) - tolerance:
                    cost = cost_three_tier(exits, c_gate, c_mid, c_top)
                    if best is None or cost < best[1]:
                        best = ((tau_gate, tau_mid), cost)
            predictions, exits = route_three_tier(gt, mt, et, *best[0])
            cost = cost_three_tier(exits, c_gate, c_mid, c_top)
            row += f"{(predictions == y_test).mean() * 100:>21.2f} {c_full / cost:>7.2f}x"
        print(row)

    rule("Table 8  global against per-language thresholds")
    for tolerance in [0.10, 0.25]:
        tau = select_tau_two_tier(gate_val, ens_val, y_val, ref_val, tolerance, c_gate, c_full)
        global_pred, global_esc = route_two_tier(gate_test, ens_test, tau)
        thresholds = select_tau_per_language(gate_val, ens_val, y_val, lang_val, tolerance)
        local_pred, local_esc = apply_per_language(gate_test, ens_test, lang_test, thresholds)
        a, b, p = mcnemar(global_pred, local_pred, y_test)
        print(f"  eps={tolerance}pp  global tau={tau:.4f} escalation {global_esc.mean() * 100:5.2f}% "
              f"acc {(global_pred == y_test).mean() * 100:.2f} "
              f"speedup {c_full / cost_two_tier(global_esc, c_gate, c_full):.2f}x")
        pretty = ' '.join(f"{k}={v:.3f}" for k, v in sorted(thresholds.items()))
        print(f"{'':>14}per-language {pretty}  escalation {local_esc.mean() * 100:5.2f}% "
              f"acc {(local_pred == y_test).mean() * 100:.2f} "
              f"speedup {c_full / cost_two_tier(local_esc, c_gate, c_full):.2f}x  "
              f"McNemar b01={a} b10={b} p={p:.3f}")

    rule("Table 9  per-language mechanism")
    print(f"{'language':<14}{'n':>7}{'gate acc':>10}{'ensemble':>10}{'headroom':>10}"
          f"{'DIS':>8}{'ECE':>9}{'escalation':>12}")
    gate_accs, escalations, headrooms, eces = [], [], [], []
    for language in LANGUAGES:
        mask = lang_test == language
        g_acc = accuracy(gate_test[mask], y_test[mask])
        e_acc = accuracy(ens_test[mask], y_test[mask])
        rate = (gate_test[mask].max(1) <= 0.845).mean() * 100
        ece = expected_calibration_error(gate_test[mask], y_test[mask])
        gate_accs.append(g_acc); escalations.append(rate)
        headrooms.append(e_acc - g_acc); eces.append(ece)
        print(f"{language:<14}{mask.sum():>7,}{g_acc:>10.2f}{e_acc:>10.2f}{e_acc - g_acc:>10.2f}"
              f"{difficulty_inversion_score(gate_test[mask], y_test[mask]):>8.2f}"
              f"{ece:>9.4f}{rate:>11.2f}%")
    for name, driver in [('gate accuracy', gate_accs), ('headroom', headrooms), ('ECE', eces)]:
        print(f"  correlation of escalation with {name:<16} "
              f"r = {np.corrcoef(driver, escalations)[0, 1]:+.4f}")

    rule("Table 10  distribution-free certificates")
    print(f"{'alpha':>8}{'tau':>8}{'val damage':>12}{'test damage':>13}{'holds':>7}"
          f"{'test acc':>10}{'speedup':>9}")
    for alpha_pp in [0.10, 0.25, 0.50, 1.00]:
        chosen = calibrate(gate_val, ens_val, y_val, alpha_pp / 100, 0.05, DEFAULT_GRID)
        if chosen is None:
            print(f"{alpha_pp:>8.2f}  nothing certified")
            continue
        tau, risk_val = chosen
        risk_test, escalated = damage_rate(gate_test, ens_test, y_test, tau)
        predictions, _ = route_two_tier(gate_test, ens_test, tau)
        cost = cost_two_tier(escalated, c_gate, c_full)
        print(f"{alpha_pp:>8.2f}{tau:>8.3f}{risk_val * 100:>11.4f}pp{risk_test * 100:>12.4f}pp"
              f"{str(risk_test * 100 <= alpha_pp):>7}"
              f"{(predictions == y_test).mean() * 100:>10.2f}{c_full / cost:>8.2f}x")
    print(f"\n  Hoeffding slack at n={len(y_val)}, delta=0.05: "
          f"{np.sqrt(np.log(1 / 0.05) / (2 * len(y_val))) * 100:.3f} pp regardless of the data")

    rule("Table 11  training-time components")
    def arm_stats(paths):
        accs = [accuracy(np.load(p, allow_pickle=True)['probs_test'], y_test) for p in paths]
        diss = [difficulty_inversion_score(np.load(p, allow_pickle=True)['probs_test'], y_test)
                for p in paths]
        return np.array(accs), np.array(diss)
    baseline = sorted(glob.glob(os.path.join(args.probs, 'probs_bilstm_ce_seed*.npz')))
    if baseline:
        base_acc, base_dis = arm_stats(baseline)
        print(f"{'arm':<24}{'n':>3}{'acc':>8}{'DIS':>8}{'sd':>7}{'d DIS':>8}{'p':>8}")
        print(f"{'cross-entropy':<24}{len(baseline):>3}{base_acc.mean():>8.2f}"
              f"{base_dis.mean():>8.2f}{base_dis.std(ddof=1):>7.2f}")
        for epsilon in [0.1, 0.3, 0.5, 0.7]:
            paths = sorted(glob.glob(os.path.join(
                args.probs, f'probs_bilstm_dar_eps{epsilon}_seed*.npz')))
            if not paths:
                continue
            acc, dis = arm_stats(paths)
            print(f"{'+ margin eps=' + str(epsilon):<24}{len(paths):>3}{acc.mean():>8.2f}"
                  f"{dis.mean():>8.2f}{dis.std(ddof=1):>7.2f}{dis.mean() - base_dis.mean():>+8.2f}"
                  f"{ttest_ind(dis, base_dis, equal_var=False).pvalue:>8.3f}")

    rule("Table 12  replication")
    hb_bundle = os.path.join(args.probs_hb, 'probs_hb_bundle.npz')
    hb_gate = os.path.join(args.probs_hb, 'probs_hb_tfidf.npz')
    if os.path.exists(hb_bundle) and os.path.exists(hb_gate):
        hb = np.load(hb_bundle, allow_pickle=True)
        hg = np.load(hb_gate, allow_pickle=True)
        hb_y = hb['y_test']
        hb_ens = product_ensemble(hb['pa_test'], hb['pb_test'])
        hb_ens_val = product_ensemble(hb['pa_val'], hb['pb_val'])
        hb_gate_test, hb_gate_val = hg['probs_test'], hg['probs_val']
        tau = select_tau_two_tier(hb_gate_val, hb_ens_val, hb['y_val'],
                                  accuracy(hb_ens_val, hb['y_val']), 0.10, c_gate, c_full)
        predictions, escalated = route_two_tier(hb_gate_test, hb_ens, tau)
        cost = cost_two_tier(escalated, c_gate, c_full)
        a, b, p = mcnemar(hb_ens.argmax(1), predictions, hb_y)
        print(f"  gate acc {accuracy(hb_gate_test, hb_y):.2f}  "
              f"ensemble {accuracy(hb_ens, hb_y):.2f}  "
              f"escalation {escalated.mean() * 100:.2f}%  "
              f"acc {(predictions == hb_y).mean() * 100:.2f}  "
              f"speedup {c_full / cost:.2f}x  McNemar p={p:.3f}")
        fit = linregress(gate_accs, escalations)
        held_out_acc = accuracy(hb_gate_test, hb_y)
        held_out_rate = (hb_gate_test.max(1) <= 0.845).mean() * 100
        predicted = fit.slope * held_out_acc + fit.intercept
        print(f"  out-of-sample: r={fit.rvalue:.4f}  predicted {predicted:.2f}%  "
              f"actual {held_out_rate:.2f}%  residual {abs(predicted - held_out_rate):.2f} pp")
    else:
        print("  replication arrays not present")

    rule("Proposition check  escalation sets under recalibration")
    for name, probs in [('gate', gate_test), ('recurrent', lstm['probs_test']),
                        ('XLM-R', bundle['pb_test']), ('MuRIL', bundle['pa_test'])]:
        temperature = fit_temperature(gate_val if name == 'gate' else bundle['pb_val'], y_val)
        print(f"  {name:<12} identical escalation sets at matched rates: "
              f"{escalation_sets_identical(probs, temperature)}")

    rule("Table 13  throughput")
    for name, ms in [('two-way ensemble', c_full),
                     ('two-tier cascade', c_gate + 0.1143 * c_full)]:
        print(f"  {name:<24}{ms:>9.4f} ms   {ms * 1e6 / 3.6e6:>7.3f} GPU-hours per million")


if __name__ == '__main__':
    main()
