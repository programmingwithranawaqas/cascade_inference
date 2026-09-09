# Data

Neither corpus is redistributed here.

## Primary corpus

86,701 news headlines in English, Urdu, Roman Urdu and code-mixed text.
Available at `doi:10.5281/zenodo.21946028`.

Place the four files in this directory, names unchanged:

```
data/mixed_english.csv
data/mixed_urdu.csv
data/mixed_roman_urdu.csv
data/mixed_multilingual.csv
```

Each file carries a `headline` column and a `label` column taking the values
`fake` and `real`.

## Replication corpus

Hook and Bait, 78,409 Urdu items, distributed as six spreadsheet shards at
https://github.com/Sheetal83/Hook-and-Bait-Urdu

Clone it beside this repository and consolidate the shards:

```bash
git clone https://github.com/Sheetal83/Hook-and-Bait-Urdu.git
python scripts/prepare_hookbait.py --src Hook-and-Bait-Urdu --out data_hb/hookbait.csv
```

The shards are sequential by record number but differ in column naming and label
encoding, so they cannot be concatenated directly. The consolidation script
normalises both and reports the duplicate and label-conflict counts.

## Expected counts

After loading, `scripts/separability_probe.py` should report 86,701 rows for the
primary corpus and 78,409 for the replication corpus, with 36,230 fake and
42,179 real in the latter. If the counts differ, stop: every downstream number
is indexed against a split built from these files.
