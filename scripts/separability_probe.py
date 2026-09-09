"""Estimate how much of a corpus is reachable by surface features alone.

A cascade exploits heterogeneity in instance difficulty, so this quantity is the
scope condition for the efficiency results. Reported on the deduplicated corpus,
since a ceiling estimated over duplicated text is inflated.

    python scripts/separability_probe.py --data_dir data
    python scripts/separability_probe.py --hookbait data_hb/hookbait.csv
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlfnd.data import SEED, deduplicate, load_corpus
from mlfnd.models import LOGREG_KWARGS, TFIDF_KWARGS


def probe(corpus, name):
    clean = deduplicate(corpus)
    print(f"{name}: {len(corpus):,} rows, {len(clean):,} after deduplication")
    train, test = train_test_split(clean, test_size=0.2, random_state=SEED,
                                   stratify=clean['label_bin'])
    vectoriser = TfidfVectorizer(**TFIDF_KWARGS)
    classifier = LogisticRegression(**LOGREG_KWARGS).fit(
        vectoriser.fit_transform(train['headline']), train['label_bin'])
    ceiling = (classifier.predict(vectoriser.transform(test['headline']))
               == test['label_bin']).mean()
    majority = max(test['label_bin'].mean(), 1 - test['label_bin'].mean())
    print(f"  lexical ceiling {ceiling * 100:.2f}   majority {majority * 100:.2f}   "
          f"lift {(ceiling - majority) * 100:+.2f} pp")
    if 'language' in clean.columns and clean['language'].nunique() > 1:
        for language in sorted(clean['language'].unique()):
            mask = test['language'] == language
            if mask.sum() == 0:
                continue
            acc = (classifier.predict(vectoriser.transform(test['headline'][mask]))
                   == test['label_bin'][mask]).mean()
            print(f"    {language:<12} {acc * 100:6.2f}   n = {mask.sum():,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--hookbait', default=None)
    args = parser.parse_args()
    if args.hookbait:
        corpus = pd.read_csv(args.hookbait)
        corpus['language'] = 'Urdu'
        probe(corpus, 'replication corpus')
    else:
        probe(load_corpus(args.data_dir), 'primary corpus')


if __name__ == '__main__':
    main()
