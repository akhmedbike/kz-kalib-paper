# Data

The canonical train/dev/test split used in the paper: 754/162/162 sentences
(7,376 / 1,566 / 1,594 syntactic tokens), seed 42, decontaminated.

- `train.conllu`, `dev.conllu`, `test.conllu` — the split itself
  (syntactic words only; multiword-token ranges are skipped by the parsers)
- `split_meta.json` — split parameters and per-split statistics
- `SHA256SUMS` — integrity checksums for the files above

## Source and license

The sentences are drawn from the Universal Dependencies **Kazakh-KTB**
treebank (<https://universaldependencies.org/treebanks/kk_ktb/index.html>)
by Aibek Makazhanov, Jonathan North Washington and Francis Tyers.

The treebank is distributed under the **Creative Commons
Attribution-ShareAlike 4.0** license. The split files in this directory are
redistributed under the same CC BY-SA 4.0 license. This applies to this
directory only; the code in this repository is MIT (see the root `LICENSE`).

If you use this data, please cite the treebank papers:

- Francis Tyers, Jonathan North Washington. *Towards a Free/Open-source
  Universal-dependency Treebank for Kazakh.* TurkLang 2015, pp. 276–289.
- Aibek Makazhanov, Gaukhar Sultangazina, Yerbol Makhambetov, Gibrat
  Yessenbayev. *Syntactic Annotation of Kazakh: Following the Universal
  Dependencies Guidelines. A report.* TurkLang 2015, pp. 338–350.
