"""Train the recurrent tier-1 alternative, optionally with difficulty-aware
regularisation.

Two stages. `--stage difficulty` labels each training instance by K-fold
leave-one-out training: an instance is easy when every fold model classifies it
correctly. `--stage train` then trains the final arms, with a matched
cross-entropy baseline.

The epoch budget is fixed and early stopping is off, so the arms differ only in
the objective.

    python scripts/train_bilstm.py --stage difficulty --data_dir data
    python scripts/train_bilstm.py --stage train --data_dir data --n_seeds 5
"""
import argparse
import os
import random
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlfnd.data import assert_aligned, build_split
from mlfnd.losses import EPSILON_GRID, LAMBDA_DAR, difficulty_margin_loss
from mlfnd.models import (BiLSTM, LSTM_BATCH_SIZE, LSTM_LR, LSTM_MAX_LEN, PAD,
                          UNK, VOCAB_SIZE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text):
    return str(text).lower().split()


def build_vocab(texts, min_freq=1, max_vocab=VOCAB_SIZE):
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    kept = [word for word, n in counts.most_common() if n >= min_freq][:max_vocab - 2]
    return {word: i + 2 for i, word in enumerate(kept)}


class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=LSTM_MAX_LEN):
        self.texts, self.labels = list(texts), list(labels)
        self.vocab, self.max_len = vocab, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = [self.vocab.get(w, UNK) for w in tokenize(self.texts[idx])][:self.max_len]
        length = max(len(ids), 1)          # packing rejects zero lengths
        ids = ids + [PAD] * (self.max_len - len(ids))
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(length, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


def train_model(train_df, vocab, seed, device, epochs, epsilon=None, is_easy=None):
    set_seed(seed)
    model = BiLSTM(len(vocab) + 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    criterion = nn.CrossEntropyLoss()
    dataset = HeadlineDataset(train_df['headline'], train_df['label_bin'], vocab)
    difficulty = torch.tensor(is_easy, dtype=torch.long) if is_easy is not None else None
    generator = torch.Generator().manual_seed(seed)
    index_loader = DataLoader(range(len(dataset)), batch_size=LSTM_BATCH_SIZE,
                              shuffle=True, generator=generator)
    for _ in range(epochs):
        model.train()
        for indices in index_loader:
            items = [dataset[i] for i in indices.tolist()]
            ids = torch.stack([x[0] for x in items]).to(device)
            lengths = torch.stack([x[1] for x in items]).to(device)
            labels = torch.stack([x[2] for x in items]).to(device)
            optimizer.zero_grad()
            logits = model(ids, lengths)
            loss = criterion(logits, labels)
            if epsilon is not None:
                batch_difficulty = difficulty[indices].to(device)
                loss = loss + LAMBDA_DAR * difficulty_margin_loss(
                    F.softmax(logits, dim=-1), batch_difficulty, epsilon)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict(model, frame, vocab, device):
    model.eval()
    loader = DataLoader(HeadlineDataset(frame['headline'], frame['label_bin'], vocab),
                        batch_size=LSTM_BATCH_SIZE * 2, shuffle=False)
    out = []
    for ids, lengths, _ in loader:
        out.append(F.softmax(model(ids.to(device), lengths.to(device)), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def stage_difficulty(args, train, vocab, device):
    n = len(train)
    correct_counts = np.zeros(n, dtype=np.int32)
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(1000 + seed)
        folds = np.array_split(rng.permutation(n), args.k_folds)
        for fold_index, held_out in enumerate(folds):
            mask = np.ones(n, dtype=bool)
            mask[held_out] = False
            fold_train = train[mask].reset_index(drop=True)
            fold_eval = train.iloc[held_out].reset_index(drop=True)
            print(f"  seed {seed} fold {fold_index + 1}/{args.k_folds}: "
                  f"{len(fold_train):,} -> {len(fold_eval):,}")
            model = train_model(fold_train, vocab, seed * 100 + fold_index,
                                device, args.label_epochs)
            probs = predict(model, fold_eval, vocab, device)
            correct_counts[held_out] += (
                probs.argmax(1) == fold_eval['label_bin'].values).astype(np.int32)

    # an instance is easy when every fold model classified it correctly. A median
    # split is degenerate here: the correct-count distribution is heavily skewed,
    # so the median equals the number of seeds and nothing lies strictly above it
    is_easy = (correct_counts >= args.n_seeds).astype(np.int8)
    if is_easy.sum() in (0, len(is_easy)):
        raise SystemExit(
            f"degenerate difficulty split: {is_easy.sum()} easy of {len(is_easy)}; "
            f"the regulariser would have no effect")
    path = os.path.join(args.out_dir, 'difficulty.npz')
    np.savez(path, counts=correct_counts, is_easy=is_easy,
             n_seeds=np.array(args.n_seeds), k_folds=np.array(args.k_folds))
    print(f"\ncounts distribution {np.bincount(correct_counts, minlength=args.n_seeds + 1)}")
    print(f"easy {is_easy.sum():,} ({is_easy.mean() * 100:.1f}%)  "
          f"difficult {(1 - is_easy).sum():,}")
    print(f"saved {path}")


def stage_train(args, train, val, test, vocab, device):
    labels_path = os.path.join(args.out_dir, 'difficulty.npz')
    saved = np.load(labels_path)
    # derive the split from the raw counts, so a change to the rule does not
    # require repeating the fold training
    is_easy = (saved['counts'] >= int(saved['n_seeds'])).astype(np.int8)
    if is_easy.sum() in (0, len(is_easy)):
        raise SystemExit("degenerate difficulty split")
    print(f"difficulty labels: {is_easy.sum():,} easy / {(1 - is_easy).sum():,} difficult\n")

    arms = [('ce', None)] + [(f'dar_eps{eps}', eps) for eps in args.epsilon]
    for name, epsilon in arms:
        for seed in range(args.n_seeds):
            out_path = os.path.join(args.out_dir, f'probs_bilstm_{name}_seed{seed}.npz')
            if os.path.exists(out_path) and not args.overwrite:
                print(f"{out_path} exists; skipping")
                continue
            print(f"=== {name}  seed {seed}  ({args.epochs} epochs) ===")
            model = train_model(train, vocab, seed, device, args.epochs,
                                epsilon=epsilon,
                                is_easy=(is_easy if epsilon is not None else None))
            probs_val = predict(model, val, vocab, device)
            probs_test = predict(model, test, vocab, device)
            val_acc = (probs_val.argmax(1) == val['label_bin'].values).mean()
            test_acc = (probs_test.argmax(1) == test['label_bin'].values).mean()
            print(f"  val {val_acc * 100:.2f}   test {test_acc * 100:.2f}")
            np.savez(out_path, probs_val=probs_val, probs_test=probs_test,
                     y_val=val['label_bin'].values, lang_val=val['language'].values,
                     y_test=test['label_bin'].values, lang_test=test['language'].values,
                     val_acc=np.array(val_acc), test_acc=np.array(test_acc),
                     epsilon=np.array(epsilon if epsilon is not None else -1.0),
                     lam=np.array(LAMBDA_DAR), seed=np.array(seed),
                     epochs=np.array(args.epochs))
            print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['difficulty', 'train'])
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--bundle', default='probs/probs_bundle.npz')
    parser.add_argument('--out_dir', default='probs')
    parser.add_argument('--k_folds', type=int, default=8)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--label_epochs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--epsilon', nargs='*', type=float, default=EPSILON_GRID)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')
    train, val, test = build_split(args.data_dir)
    print(f"train {len(train):,}  val {len(val):,}  test {len(test):,}")
    assert_aligned(args.bundle, val, test)
    vocab = build_vocab(train['headline'])
    print(f"vocabulary {len(vocab) + 2:,}")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.stage == 'difficulty':
        stage_difficulty(args, train, vocab, device)
    else:
        stage_train(args, train, val, test, vocab, device)


if __name__ == '__main__':
    main()
