"""Consolidate the replication corpus shards into a single file.

    python scripts/prepare_hookbait.py --src Hook-and-Bait-Urdu --out data_hb/hookbait.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlfnd.data import HB_EXPECTED, deduplicate, load_hookbait_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='Hook-and-Bait-Urdu')
    parser.add_argument('--out', default='data_hb/hookbait.csv')
    parser.add_argument('--dedup', action='store_true')
    args = parser.parse_args()

    corpus = load_hookbait_shards(args.src)
    n_fake = int((corpus['label_bin'] == 1).sum())
    n_real = int((corpus['label_bin'] == 0).sum())
    print(f"loaded {len(corpus):,} rows: {n_fake:,} fake (expected {HB_EXPECTED['fake']:,}), "
          f"{n_real:,} real (expected {HB_EXPECTED['real']:,})")

    key = corpus['headline'].str.lower().str.strip()
    duplicates = int(key.duplicated().sum())
    conflicts = int((corpus.assign(_k=key).groupby('_k')['label_bin'].nunique() > 1).sum())
    print(f"exact duplicates {duplicates:,}   headlines with both labels {conflicts:,}")
    print("state the deduplication policy alongside any figure computed from this corpus")

    if args.dedup:
        corpus = deduplicate(corpus)
        print(f"after deduplication: {len(corpus):,}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    corpus[['headline', 'label_bin', 'language', 'source_file']].to_csv(
        args.out, index=False, encoding='utf-8')
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
