# Datasets

Where the image data for this project lives, and how to restore it.

## In-repo — already version controlled

`MLImages/` **is tracked in git** (11,199 files, ~136 MB) and is on the GitHub
remote. Cloning the repo gets you this data; nothing extra to download.

| Folder | Images | Size | Referenced by |
|---|---|---|---|
| `MLImages/All224/` | 3,558 jpg | 27 MB | — (224×224 set) |
| `MLImages/All64/`  | 4,758 jpg | 19 MB | `gan_generator.py` → `metadata_path` |
| `MLImages/tempAll/`| 2,868 jpg | 89 MB | `gan_generator.py` → `image_dir` |

### Manifest CSVs — force-added

`.gitignore` carries a blanket `*.csv` rule, so `MLImages/All224/metadata.csv`
and `MLImages/All64/metadata.csv` (199 KB each, 4,758 rows each) were untracked
and would have been lost when this folder was deleted. Both are now committed
via `git add -f`.

Note what they are: **single-column filename manifests**, header `Image name`,
one filename per row. They contain **no labels**. They pin the exact image
inventory and nothing more.

## Out-of-repo — NOT version controlled

A sibling folder `../MLImages/` (i.e. `~/Documents/Repos/MLImages/`) held the
original downloads. It was **never a git repo** and is being deleted to reclaim
disk. Contents, for the record:

### `OriginalZips/archive.zip` (30 MB)

Extracts to two folders with **identical contents** (2,838 jpg each — one is a
redundant copy):

- `archive/Diagnosis of Diabetic Retinopathy/`
- `archive/retino/`

Both use the same binary-label split layout:

```
<root>/
  train/  DR=1050   No_DR=1026
  test/   DR= 113   No_DR= 118
  valid/  DR= 245   No_DR= 286
```

**Source:** https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy
(the folder name matches this dataset exactly). Re-downloadable at any time —
which is why the zips are not committed here.

### `OriginalZips/OLIVES_Dataset-main.zip` (219 KB)

A source-code snapshot of the OLIVES dataset benchmark repo ("Biomarker
Interpretation with Contrastive Learning"), **not** image data. Public on GitHub.

## Known path mismatch

`cnn.py` and `svm.py` both point at:

```python
image_dir     = '../MLImages/Messidor1/All'
metadata_path = '../MLImages/Messidor1/Annotation_Base_CSV/Annotation_All.csv'
```

**No `Messidor1/` directory exists in either location** — not in the tracked
in-repo `MLImages/`, and not in the sibling folder. The Messidor-1 data was
never checked in and is not present locally. These two scripts will not run
without re-sourcing that dataset (Messidor-1 is distributed by ADCIS).

This is also where the **labels** were meant to come from: `cnn.py` reads a
`Retinopathy grade` column out of `Annotation_All.csv` (and raises if it is
absent). The in-repo manifest CSVs described above do **not** carry that column
— they are filename lists only. So the tracked images have no accompanying
grades in this repo; re-sourcing Messidor-1, or another labelled source, is
required to train.

`gan_generator.py` is likewise in a mid-experiment state — `image_dir` points at
`./MLImages/tempAll` while `metadata_path` still points at
`./MLImages/All64/metadata.csv`. Both of those paths do resolve against the
tracked in-repo data, but the pairing is probably not what you want; check it
before running.
