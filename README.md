# AI Hardware Patent Corpus

Code, vocabularies, and human validation labels for:

> **Measuring AI Hardware Innovation: Corpus Construction, Validation, and
> Measurement Bias**

---

## What this is

A validated method for constructing an AI hardware patent corpus from a broad
CPC classification sweep, and evidence that how the corpus is built
systematically changes the technology measurements obtained from it.

**Headline results**

| | |
|---|---|
| Raw records retrieved | 3,652,164 |
| Deduplicated U.S. families | 1,208,932 |
| Retained AI hardware families | **7,464** (0.62%) |
| Corpus precision (200 human labels) | **81.7%** [73.8-87.6] |
| Held-out precision (45 records) | 80.0% [66.2-89.1] |
| Inter-annotator agreement (50 records) | Cohen's kappa 0.919 |
| False negatives at the decision boundary | 1.7% |
| Prevalence ratio vs. unfiltered sweep | 2.0x (CPU) to 83.5x (neuromorphic) |

The unfiltered sweep shows no temporal trend for any architecture, while the
validated corpus shows changes of up to 16 percentage points.

---

## The rule

A patent family is retained if and only if:

1. it carries a CPC code under **G06N3/06** (physical realisation of neural
   networks), **and**
2. its title or abstract contains at least one term from the hardware, AI, or
   AI-mechanism vocabularies (`config.py`), **and**
3. its title does not indicate an out-of-scope subject, meaning quantum
   computing or machine learning applied *to* chip design and fabrication.

The intuitive alternative, which conjoins AI evidence and hardware evidence
where either may be satisfied by classification or text, achieves **42%
precision** on the same labels. `test_rule.py` documents why, with 11
hand-labelled fixtures.

---

## Repository contents

```
config.py                 all vocabularies, CPC prefixes, hyperparameters
test_rule.py              11 regression fixtures for the corpus rule

stage0_inventory.py       audit the raw export (schemas, dates, overlap)
stage1_corpus.py          ingest, deduplicate, apply the rule
stage2_embed.py           embed the corpus (PatentSBERTa)
stage2b_embed_sweep.py    embed the full candidate set
stage2c_expand.py         semantic expansion scoring (reported, not applied)
stage3_cluster.py         multi-view clustering and ablations
stage4_analysis.py        architecture shares, sweep vs. validated comparison

sweep_hdbscan.py          HDBSCAN parameter sweep
rep_sweep.py              representation sweep (granularity, view weights)
variant_eval.py           score rule variants against human labels
blind_labels2.py          blinded labelling, merge, inter-annotator agreement
arch_map.py               architecture assignment validation

make_fig_ratio.py         Fig. 1, measurement bias by corpus construction
make_fig_trajectory.py    Fig. 2, architecture trajectories
make_fig_clusters.py      cluster map and cluster_names.csv

LABELLING_GUIDE.md        the annotation instrument

labels/
  validation_labels.csv             200 adjudicated labels
  validation_labels_annotator2.csv  50 second-annotator labels

manifests/
  stage1.json ... stage4.json       per-stage config, environment, runtime
  attrition.csv                     corpus reduction, raw rows to final
  ablation.csv                      view weight ablation
```

**Not included:** the patent records themselves. Publication numbers are public
facts, but the enriched bibliographic records are subject to Lens.org terms. We
release identifiers and derived labels. The corpus can be reconstructed by
anyone with Lens.org access using the query and rule documented here.

---

## Reproducing

```bash
pip install pandas numpy pyarrow scikit-learn umap-learn \
            sentence-transformers torch matplotlib

# edit config.py: RAW_APP_DIR, RAW_GRANT_DIR, ROOT
python test_rule.py                     # expect 11/11
python stage0_inventory.py --app-dir <filing> --grant-dir <grant> --deep
python stage1_corpus.py --refresh       # ~30 min
python stage2_embed.py                  # ~1 min
python stage3_cluster.py --ablate       # ~15 min
python stage3_cluster.py
python stage4_analysis.py
python stage4_analysis.py --year-col pub_year
```

All runs use `SEED = 42`. Each stage writes a `manifest.json` recording its
configuration, environment, and runtime. The manifests in this repository are
from the run reported in the paper.

**Source query.** Lens.org, CPC and IPC subclasses G06F, G06T, G06V, G06K,
G06N and G06Q, with a date filter and no jurisdiction restriction. Two exports,
one sliced by filing date and one by publication date. Retrieved Jun. 1, 2026.

---

## Validation labels

`labels/validation_labels.csv` contains 200 records, stratified 60% retained,
30% near-miss exclusions and 10% far exclusions, labelled blind. The annotator
saw only title, abstract and CPC codes, not whether the rule had retained the
record.

| Column | |
|---|---|
| `publication_number` | join key |
| `side` | retained / excluded_near / excluded_far |
| `stage` | development (rows 1-120) or held-out (121-200) |
| `is_ai_hardware` | Y / N / BORDERLINE |
| `architecture` | annotator's free-text description |

Labelling was performed by one annotator, a doctoral researcher working on AI
hardware patent analytics. A second annotator independently labelled 50 of the
same records using the same instrument, without sight of the first annotator's
labels; raw agreement was 96.0% and Cohen's kappa was 0.919. Those labels are
in `labels/validation_labels_annotator2.csv`. The instrument itself is in
`LABELLING_GUIDE.md`.

---

## Three failure modes worth knowing about

Each produced plausible output and was invisible in summary statistics.

**Substring matching without word boundaries.** `tpu` matches "ou**tpu**t",
`npu` matches "i**npu**t", and `asic` matches "b**asic**". Since nearly every
patent abstract contains *input* and *output*, two of six architecture series
were measuring nothing.

**Priority chains listed newest-first.** Lens.org orders priority numbers
newest to oldest, so the earliest priority is the *last* element. Taking the
first, which is the natural implementation, assigns every continuation its own
family. This affected 981,378 records, or 66%; correcting it removed 274,551
duplicate families.

**Classification codes as evidence of subject matter.** Most G06N subgroups
denote software: G06N3/08 covers learning methods and G06N20/00 machine
learning generally. Hardware codes such as G06F9/38 and G06F12 are assigned for
incidental implementation aspects of software inventions. Both conditions of
the intuitive rule were routinely satisfied without either being true.

---

## Citation

```bibtex
@inproceedings{aihw2026,
  title     = {Measuring AI Hardware Innovation: Corpus Construction,
               Validation, and Measurement Bias},
  author    = {Khaing Phyo Wai and Pao-Li Chang},
  booktitle = {IEEE International Conference on Big Data},
  year      = {2026},
  note      = {Under review}
}
```

Patent data retrieved from The Lens (https://www.lens.org).

## Licence

Code is released for academic use. Labels and derived data may be reused with
attribution. Neither covers the underlying Lens.org records, which are subject
to that provider's terms.
