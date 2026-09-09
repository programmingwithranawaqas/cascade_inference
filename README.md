# Cost-aware cascaded inference for multilingual misinformation detection

Code and analysis for *When Calibration Cannot Help: An Invariance Result for
Confidence-Gated Cascades, and an Audit of Three Cascade Methods*.

A confidence-gated cascade classifies most inputs with a cheap model and
escalates only low-confidence inputs to expensive ones. This repository contains
the cascade implementation, reimplementations of three published cascade methods
including their training-time components, the cost-measurement harness, and one
script that regenerates every table and figure in the paper from the saved
outputs.

## What is here

```
mlfnd/
  data.py           corpus loading, the canonical split, the replication corpus
  metrics.py        confidence, calibration error, rank statistics, paired tests
  cascade.py        routing, cost accounting, threshold selection
  certificates.py   distribution-free threshold calibration
  models.py         model definitions and the shared training configuration
  losses.py         training-time objectives from the two cascade baselines
scripts/
  train_gate.py            fit the tier-1 gate
  train_bilstm.py          recurrent tier-1, with and without margin regularisation
  train_transformers.py    fine-tune the transformer tiers, with and without logit norm
  train_hookbait.py        the same protocol on the replication corpus
  prepare_hookbait.py      consolidate the replication corpus shards
  separability_probe.py    lexical ceiling against the majority baseline
  benchmark_cost.py        wall-clock cost, all tiers, one session
  make_tables.py           regenerate every table
  make_figures.py          regenerate every figure
```

## Data

Neither corpus is redistributed here. See `data/README.md` for how to obtain
both and where to place them.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# install the torch build matching your CUDA version, for example
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Reproducing the results

Every script asserts that the reconstructed split matches the saved probability
arrays before doing any work, and exits if it does not. Run that check first:

```bash
python -c "from mlfnd.data import build_split, assert_aligned; \
tr, va, te = build_split('data'); assert_aligned('probs/probs_bundle.npz', va, te)"
```

Then, in order:

```bash
# tier-1 gate and the corpus separability probe
python scripts/train_gate.py --data_dir data
python scripts/separability_probe.py --data_dir data

# transformer tiers
python scripts/train_transformers.py --model muril --arm baseline
python scripts/train_transformers.py --model xlmr  --arm baseline

# recurrent tier-1 alternative and the margin-regularised arms
python scripts/train_bilstm.py --stage difficulty --data_dir data
python scripts/train_bilstm.py --stage train --data_dir data --n_seeds 5

# logit-normalised arms, matched on training length
for seed in 0 1 2 3 4; do
  python scripts/train_transformers.py --model muril --arm baseline \
      --fixed_epochs 6 --seed $seed
  python scripts/train_transformers.py --model muril --arm logitnorm --kappa 0.04 \
      --fixed_epochs 6 --seed $seed
done

# replication corpus
python scripts/prepare_hookbait.py --src Hook-and-Bait-Urdu --out data_hb/hookbait.csv
python scripts/train_hookbait.py --model gate
python scripts/train_hookbait.py --model muril
python scripts/train_hookbait.py --model xlmr
python scripts/train_hookbait.py --bundle

# cost measurement: pin the CPU threads before the process starts
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/benchmark_cost.py --data_dir data --out results/costs.json

# tables and figures
python scripts/make_tables.py --probs probs --costs results/costs.json
python scripts/make_figures.py --probs probs --costs results/costs.json
```

## Notes on the experimental protocol

Several choices below look like implementation detail but change the results,
and are recorded here so they are not silently reverted.

**The stratum key is built from the label strings, not the binarised label.**
Both encode the same strata, but the sorted order of the keys differs, and the
integer form produces a partition of identical sizes sharing only about 20% of
its test set with the one used throughout.

**Rank statistics are computed on the logit scale.** Confidence saturates as a
function of the logit, so a rank statistic computed on probabilities is
contaminated by ties at high confidence. One ensemble here reads 77.36 on the
probability scale and 95.28 on the logit scale.

**Cost is measured over several random batches per configuration.** Per-instance
cost depends on the texts in a batch as well as its size, because the recurrent
model packs sequences and vectorisation scales with text length. Timing a single
batch per size confounds the batch-size effect with that batch's composition; two
such runs differing only in the sample gave a six-fold difference at batch 16.

**Comparisons between training objectives fix the epoch budget.** With early
stopping the arms train for different numbers of epochs and the difference in
training length is attributed to the objective. Use `--fixed_epochs`.

**Positive effects are reported at five seeds.** Two effects here changed
materially between three and five seeds, and one changed sign.

**Thresholds are selected on validation and the test split is used once**, for
the selected configuration.

## Cost figures

Every cost number derives from a single JSON file written by
`benchmark_cost.py`, which records the device, driver, library versions,
precision flags and thread counts. Report the batch size, the padding regime and
the timing regime alongside any speedup; without them the figure is not
reproducible.

## Licence

MIT, see `LICENSE`.
