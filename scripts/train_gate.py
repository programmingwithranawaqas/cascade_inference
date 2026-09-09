"""Fit the tier-1 TF-IDF gate and save its probabilities.

    python scripts/train_gate.py --data_dir data --out probs/probs_tfidf.npz
"""
import argparse
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlfnd.data import assert_aligned, build_split
from mlfnd.models import LOGREG_KWARGS, TFIDF_KWARGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--bundle', default='probs/probs_bundle.npz')
    parser.add_argument('--out', default='probs/probs_tfidf.npz')
    args = parser.parse_args()

    train, val, test = build_split(args.data_dir)
    print(f"train {len(train):,}  val {len(val):,}  test {len(test):,}")
    assert_aligned(args.bundle, val, test)

    vectoriser = TfidfVectorizer(**TFIDF_KWARGS)
    features = vectoriser.fit_transform(train['headline'].astype(str))
    classifier = LogisticRegression(**LOGREG_KWARGS).fit(features, train['label_bin'])
    print(f"features: {features.shape[1]:,}")

    probs_val = classifier.predict_proba(
        vectoriser.transform(val['headline'].astype(str))).astype(np.float32)
    probs_test = classifier.predict_proba(
        vectoriser.transform(test['headline'].astype(str))).astype(np.float32)
    val_acc = (probs_val.argmax(1) == val['label_bin'].values).mean()
    test_acc = (probs_test.argmax(1) == test['label_bin'].values).mean()
    print(f"val {val_acc * 100:.2f}   test {test_acc * 100:.2f}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out, probs_val=probs_val, probs_test=probs_test,
             y_val=val['label_bin'].values, lang_val=val['language'].values,
             y_test=test['label_bin'].values, lang_test=test['language'].values,
             val_acc=np.array(val_acc), test_acc=np.array(test_acc),
             config=np.array(f"{TFIDF_KWARGS} + {LOGREG_KWARGS}"))
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
