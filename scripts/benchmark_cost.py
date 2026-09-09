"""Measure wall-clock inference cost for every tier in one session.

Four properties of the protocol matter for the comparison to mean anything.

Warm-up. The first forward passes include context creation, kernel autotuning
and lazy loading, which inflate whichever model runs first. Those iterations are
discarded.

Tokenisation outside the timed region. Two regimes are reported separately:
model-only, with inputs already resident on the device, and end-to-end,
including tokenisation and transfer. Iterating a data loader that tokenises on
the fly bills tokenisation as model cost and favours whichever model has the
cheaper tokeniser.

Batch composition, not only batch size. Per-instance cost depends on the texts in
a batch as well as its size: the recurrent model packs sequences and so runs for
the longest true length even under fixed padding, and vectorisation scales with
text length. Timing one batch per size confounds batching with the composition of
that batch. Several independent random batches are drawn per configuration and
the median taken across batches and repetitions, which estimates expected cost
over the test distribution.

Provenance. Device, driver, library versions, precision flags and thread counts
are written into the output, since a cost figure without them is not reproducible.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python scripts/benchmark_cost.py --data_dir data --out results/costs.json
"""
import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlfnd.data import build_split
from mlfnd.models import HF_NAMES, LOGREG_KWARGS, LSTM_MAX_LEN, MAX_LEN, TFIDF_KWARGS


def synchronise(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize()


def time_calls(fn, n_warmup, n_repeat, device=None):
    for _ in range(n_warmup):
        fn()
    if device is not None:
        synchronise(device)
    timings = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        fn()
        if device is not None:
            synchronise(device)
        timings.append(time.perf_counter() - start)
    return timings


def summarise(timings, batch_size):
    per_instance = sorted(t / batch_size * 1000 for t in timings)
    quantile = lambda q: per_instance[min(len(per_instance) - 1,
                                          int(round(q * (len(per_instance) - 1))))]
    return {
        'median_ms_per_instance': statistics.median(per_instance),
        'p25_ms_per_instance': quantile(0.25),
        'p75_ms_per_instance': quantile(0.75),
        'p95_ms_per_instance': quantile(0.95),
        'min_ms_per_instance': per_instance[0],
        'max_ms_per_instance': per_instance[-1],
        'iqr_ms_per_instance': quantile(0.75) - quantile(0.25),
        'n_timings': len(per_instance),
    }


def sample_batches(texts, batch_sizes, seed, n_batches):
    """Independent draws per batch size, so batches are neither nested nor tied
    to a single composition."""
    rng = np.random.default_rng(seed)
    return {bs: [[texts[i] for i in rng.choice(len(texts), size=bs, replace=False)]
                 for _ in range(n_batches)]
            for bs in batch_sizes}


def provenance(args):
    info = {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'warmup': args.warmup,
        'repeats': args.repeats,
        'n_batches': args.n_batches,
        'batch_seed': args.seed,
        'env_threads': {k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
    }
    try:
        import torch
        info.update({
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'matmul_precision': torch.get_float32_matmul_precision(),
        })
        if torch.cuda.is_available():
            info['device'] = torch.cuda.get_device_name(0)
            info['capability'] = '.'.join(map(str, torch.cuda.get_device_capability(0)))
        else:
            info['device'] = 'cpu'
    except ImportError:
        info['device'] = 'cpu'
    try:
        info['nvidia_smi'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version,clocks.max.sm,power.limit',
             '--format=csv,noheader'], text=True).strip()
    except Exception:
        info['nvidia_smi'] = None
    return info


def bench_gate(train_texts, train_labels, batches, batch_sizes, args):
    """The gate never touches the accelerator, so it is timed on one CPU thread."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    print("\n=== tier-1 gate (CPU) ===")
    vectoriser = TfidfVectorizer(**TFIDF_KWARGS)
    features = vectoriser.fit_transform(train_texts)
    classifier = LogisticRegression(**LOGREG_KWARGS).fit(features, train_labels)
    print(f"features {features.shape[1]:,}")
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
    except ImportError:
        limiter = None
        print("  threadpoolctl unavailable; set OMP_NUM_THREADS=1 in the environment")

    result = {'n_features': int(features.shape[1]), 'by_batch': {}}
    for bs in batch_sizes:
        model_only, end_to_end, lengths = [], [], []
        for batch in batches[bs]:
            encoded = vectoriser.transform(batch)
            lengths += [len(t.split()) for t in batch]
            model_only += time_calls(lambda: classifier.predict_proba(encoded),
                                     args.warmup, args.repeats)
            end_to_end += time_calls(
                lambda b=batch: classifier.predict_proba(vectoriser.transform(b)),
                args.warmup, args.repeats)
        result['by_batch'][bs] = {
            'model_only': summarise(model_only, bs),
            'end_to_end': summarise(end_to_end, bs),
            'n_batches': len(batches[bs]),
            'mean_true_len': float(np.mean(lengths)),
        }
        print(f"  bs={bs:<4} model-only {result['by_batch'][bs]['model_only']['median_ms_per_instance']:8.4f}"
              f"   end-to-end {result['by_batch'][bs]['end_to_end']['median_ms_per_instance']:8.4f} ms")
    if limiter is not None:
        limiter.__exit__(None, None, None)
    return result


def bench_transformer(key, batches, batch_sizes, args, device, padding):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = HF_NAMES[key]
    print(f"\n=== {name} [{padding} padding] ===")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=2, use_safetensors=True).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters {n_params:,}")
    pad_kwargs = (dict(padding='max_length', max_length=MAX_LEN) if padding == 'fixed'
                  else dict(padding=True, max_length=MAX_LEN))

    result = {'n_params': n_params, 'max_len': MAX_LEN, 'padding': padding, 'by_batch': {}}
    for bs in batch_sizes:
        model_only, end_to_end, widths, true_lengths = [], [], [], []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for batch in batches[bs]:
            encoded = tokenizer(batch, truncation=True, return_tensors='pt', **pad_kwargs)
            on_device = {k: v.to(device) for k, v in encoded.items()}
            widths.append(int(encoded['input_ids'].shape[1]))
            true_lengths += [len(t) for t in tokenizer(
                batch, truncation=True, max_length=MAX_LEN)['input_ids']]
            with torch.no_grad():
                model_only += time_calls(lambda: model(**on_device),
                                         args.warmup, args.repeats, device)

                def end_to_end_call(b=batch):
                    enc = tokenizer(b, truncation=True, return_tensors='pt', **pad_kwargs)
                    model(**{k: v.to(device) for k, v in enc.items()})

                end_to_end += time_calls(end_to_end_call, args.warmup, args.repeats, device)
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 2
                if device.type == 'cuda' else None)
        result['by_batch'][bs] = {
            'model_only': summarise(model_only, bs),
            'end_to_end': summarise(end_to_end, bs),
            'peak_mem_mb': peak, 'n_batches': len(batches[bs]),
            'padded_len_median': int(np.median(widths)),
            'mean_true_len': float(np.mean(true_lengths)),
        }
        print(f"  bs={bs:<4} model-only {result['by_batch'][bs]['model_only']['median_ms_per_instance']:8.4f}"
              f"   end-to-end {result['by_batch'][bs]['end_to_end']['median_ms_per_instance']:8.4f} ms"
              f"   padded~{result['by_batch'][bs]['padded_len_median']}")
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def bench_bilstm(batches, vocab, batch_sizes, args, device, padding):
    import torch
    from mlfnd.models import BiLSTM, PAD, UNK
    from scripts.train_bilstm import tokenize
    print(f"\n=== recurrent tier-1 [{padding} padding] ===")
    model = BiLSTM(len(vocab) + 2).to(device).eval()
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}")

    def encode(batch):
        sequences = [[vocab.get(w, UNK) for w in tokenize(t)][:LSTM_MAX_LEN] for t in batch]
        lengths = [max(len(s), 1) for s in sequences]
        width = LSTM_MAX_LEN if padding == 'fixed' else max(lengths)
        ids = [s + [PAD] * (width - len(s)) for s in sequences]
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(lengths, dtype=torch.long), width)

    result = {'max_len': LSTM_MAX_LEN, 'padding': padding, 'by_batch': {}}
    for bs in batch_sizes:
        model_only, end_to_end, widths, true_lengths = [], [], [], []
        for batch in batches[bs]:
            ids, lengths, width = encode(batch)
            ids_gpu, lengths_gpu = ids.to(device), lengths.to(device)
            widths.append(width)
            true_lengths += list(lengths.numpy())
            with torch.no_grad():
                model_only += time_calls(lambda: model(ids_gpu, lengths_gpu),
                                         args.warmup, args.repeats, device)

                def end_to_end_call(b=batch):
                    i, l, _ = encode(b)
                    model(i.to(device), l.to(device))

                end_to_end += time_calls(end_to_end_call, args.warmup, args.repeats, device)
        result['by_batch'][bs] = {
            'model_only': summarise(model_only, bs),
            'end_to_end': summarise(end_to_end, bs),
            'n_batches': len(batches[bs]),
            'padded_len_median': int(np.median(widths)),
            'mean_true_len': float(np.mean(true_lengths)),
        }
        print(f"  bs={bs:<4} model-only {result['by_batch'][bs]['model_only']['median_ms_per_instance']:8.4f}"
              f"   end-to-end {result['by_batch'][bs]['end_to_end']['median_ms_per_instance']:8.4f} ms")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--models', nargs='+', default=['gate', 'bilstm', 'xlmr', 'muril'])
    parser.add_argument('--batch_sizes', nargs='+', type=int,
                        default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument('--padding', nargs='+', default=['fixed', 'dynamic'])
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--n_batches', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='results/costs.json')
    args = parser.parse_args()

    import torch
    from scripts.train_bilstm import build_vocab
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = provenance(args)
    print(f"device {info['device']}   torch {info.get('torch')}   "
          f"tf32 matmul {info.get('matmul_allow_tf32')}")

    train, _, test = build_split(args.data_dir)
    texts = [str(t) for t in test['headline'].tolist()]
    if len(texts) < max(args.batch_sizes):
        raise SystemExit("not enough test rows for the requested batch sizes")
    vocab = build_vocab(train['headline'])
    batches = sample_batches(texts, args.batch_sizes, args.seed, args.n_batches)
    print(f"{args.n_batches} random batches per configuration, seed {args.seed}")

    results = {'provenance': info, 'batch_sizes': args.batch_sizes, 'models': {}}
    if 'gate' in args.models:
        results['models']['gate'] = {'fixed': bench_gate(
            train['headline'].astype(str).tolist(), train['label_bin'].values,
            batches, args.batch_sizes, args)}
    for padding in args.padding:
        for key in args.models:
            if key == 'gate':
                continue
            measured = (bench_bilstm(batches, vocab, args.batch_sizes, args, device, padding)
                        if key == 'bilstm'
                        else bench_transformer(key, batches, args.batch_sizes, args,
                                               device, padding))
            results['models'].setdefault(key, {})[padding] = measured

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(results, handle, indent=2, default=float)
    print(f"\nsaved {args.out}")
    print("report the batch size, padding regime and timing regime with every cost figure")


if __name__ == '__main__':
    main()
