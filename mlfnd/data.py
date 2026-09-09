"""Corpus loading and the canonical train/validation/test partition.

The partition here is the one every result in the paper is computed on. It is
reproduced by `build_split` and asserted against the saved probability arrays by
`assert_aligned` before any experiment is run.

A note on the stratum key. It is built from the original label strings, not from
the binarised integer label. Both encode the same strata, but `train_test_split`
allocates indices per stratum in the sorted order of the stratum keys, and
'fake' < 'real' is the reverse of 0 (real) < 1 (fake). Using the integer form
yields a partition of identical sizes that shares only about 20% of its test set
with the one used here.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

FILE_MAP = {
    'English': 'mixed_english.csv',
    'Urdu': 'mixed_urdu.csv',
    'Roman_Urdu': 'mixed_roman_urdu.csv',
    'Code_Mixed': 'mixed_multilingual.csv',
}

LANGUAGES = ['Code_Mixed', 'English', 'Roman_Urdu', 'Urdu']


def load_corpus(data_dir):
    """Concatenate the four language files into one frame with a binary label."""
    frames = []
    for language, filename in FILE_MAP.items():
        frame = pd.read_csv(os.path.join(data_dir, filename))
        frame['language'] = language
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True).dropna(subset=['headline', 'label'])
    full['label_bin'] = (full['label'].str.strip().str.lower() == 'fake').astype(int)
    full['strata'] = full['language'] + '_' + full['label']
    return full


def build_split(data_dir, seed=SEED):
    """The 70/10/20 partition: 60,690 / 8,670 / 17,341."""
    np.random.seed(seed)
    full = load_corpus(data_dir)
    train_val, test = train_test_split(
        full, test_size=0.20, random_state=seed, stratify=full['strata'])
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=seed, stratify=train_val['strata'])
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def deduplicate(frame):
    """Drop exact duplicate headlines and headlines carrying conflicting labels.

    Applied to the separability probe only. The cascade experiments use the full
    split, since a deployed system receives duplicates.
    """
    key = frame['headline'].astype(str).str.lower().str.strip()
    tagged = frame.assign(_key=key)
    consistent = tagged.groupby('_key')['label_bin'].transform('nunique') == 1
    return tagged[consistent].drop_duplicates('_key').drop(columns='_key').reset_index(drop=True)


def assert_aligned(bundle_path, val, test):
    """Fail loudly if the reconstructed split does not match the saved arrays.

    Every downstream number is indexed against these arrays, so a silent
    mismatch would invalidate all of them.
    """
    if not os.path.exists(bundle_path):
        print(f"warning: {bundle_path} not found, alignment not verified")
        return False
    saved = np.load(bundle_path, allow_pickle=True)
    checks = [
        (saved['y_val'], val['label_bin'].values, 'y_val'),
        (saved['lang_val'].astype(str), val['language'].values.astype(str), 'lang_val'),
        (saved['y_test'], test['label_bin'].values, 'y_test'),
        (saved['lang_test'].astype(str), test['language'].values.astype(str), 'lang_test'),
    ]
    mismatched = [name for saved_col, new_col, name in checks
                  if len(saved_col) != len(new_col) or not np.array_equal(saved_col, new_col)]
    if mismatched:
        raise SystemExit(f"split does not match {bundle_path} on {mismatched}")
    print(f"alignment against {os.path.basename(bundle_path)}: ok")
    return True


# --------------------------------------------------------------- replication corpus

HB_EXPECTED = {'fake': 36_230, 'real': 42_179, 'total': 78_409}


def _normalise_hb_label(value):
    """The shards encode the label as 'Fake', 'FAKE' or a Python boolean."""
    if isinstance(value, bool):
        return 0 if value else 1
    text = str(value).strip().lower()
    if text in ('fake', 'false', '0'):
        return 1
    if text in ('true', 'real', '1'):
        return 0
    return None


def load_hookbait_shards(src_dir):
    """Consolidate the six spreadsheet shards of the replication corpus.

    The shards are sequential by record number but differ in column naming
    (case and trailing whitespace) and in label encoding, so they cannot be
    concatenated directly.
    """
    import glob
    frames = []
    for path in sorted(glob.glob(os.path.join(src_dir, '*.xlsx'))):
        for sheet in pd.ExcelFile(path).sheet_names:
            sheet_frame = pd.read_excel(path, sheet_name=sheet)
            sheet_frame = sheet_frame.rename(
                columns={c: str(c).strip().lower() for c in sheet_frame.columns})
            text_col = next((c for c in sheet_frame.columns
                             if 'news item' in c or c == 'news'), None)
            label_col = next((c for c in sheet_frame.columns if c == 'label'), None)
            if text_col is None or label_col is None:
                print(f"skipping {os.path.basename(path)} [{sheet}]: no text/label column")
                continue
            frames.append(pd.DataFrame({
                'headline': sheet_frame[text_col].astype(str).str.strip(),
                'label_bin': sheet_frame[label_col].map(_normalise_hb_label),
                'source_file': os.path.basename(path),
            }))
    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus[corpus['label_bin'].notna()]
    corpus = corpus[corpus['headline'].str.len() >= 3]
    corpus['label_bin'] = corpus['label_bin'].astype(int)
    corpus['language'] = 'Urdu'
    return corpus.reset_index(drop=True)


def build_hookbait_split(csv_path, seed=SEED, dedup=False):
    """The same 70/10/20 procedure, stratified on label alone: the corpus is
    monolingual, so there is no language-by-label stratum."""
    corpus = pd.read_csv(csv_path).dropna(subset=['headline', 'label_bin'])
    corpus['headline'] = corpus['headline'].astype(str)
    corpus['label_bin'] = corpus['label_bin'].astype(int)
    corpus['language'] = 'Urdu'
    if dedup:
        corpus = deduplicate(corpus)
    train_val, test = train_test_split(
        corpus, test_size=0.20, random_state=seed, stratify=corpus['label_bin'])
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=seed, stratify=train_val['label_bin'])
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))
