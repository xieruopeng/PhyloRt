# Examples

This directory contains runnable datasets that mirror the real-outbreak
applications described in the manuscript.

- `examples/flub_China/`: influenza B/Victoria (clade 3a1) in China (2020-2021),
  a **single monophyletic clade** example with **constant sampling proportion**
  (simpler use case). Link: [https://doi.org/10.1093/ve/veac062](https://doi.org/10.1093/ve/veac062).
- `examples/COVID_UK/`: SARS-CoV-2 local transmission lineages in the United
  Kingdom (early 2020), a **multiple monophyletic clades (local transmissions)**
  example with **time-varying sampling proportion** (more complex use case).
  Link: [https://www.science.org/doi/10.1126/science.abf2946](https://www.science.org/doi/10.1126/science.abf2946).

Each command below runs the full PhyloRt workflow and writes outputs to the
example-specific `results/` folder.

## FLUB China Example

```bash
PhyloRt \
  --tree examples/flub_China/vic_China_3a1_sub_HA.afa.treefile \
  --date-file examples/flub_China/3a1_date.tsv \
  --sampling 0.15 \
  --pre-model FLUB \
  --out-dir examples/flub_China/results \
  --n-replicates 3
```

## COVID UK Example

```bash
PhyloRt \
  --tree examples/COVID_UK/pruned_trees.nwk \
  --date-file examples/COVID_UK/dates.tsv \
  --sampling 0.0088888889,0.001,0.0151862970,0.0186004424,0.0058315913,0.0018500443,0.0031353438,0.0085134335,0.0110609924,0.0136736516,0.0151642107,0.0170666067,0.0373456071 \
  --sampling-times 2020.1257,2020.1448,2020.1639,2020.1831,2020.2022,2020.2213,2020.2404,2020.2596,2020.2787,2020.2978,2020.3170,2020.3361 \
  --pre-model COVID \
  --out-dir examples/COVID_UK/results \
  --n-replicates 3
```

Because polytomies are randomly resolved during preprocessing, repeated runs can
produce slightly different estimates. The `--n-replicates` parameter runs
multiple stochastic replicates and combines the results, which improves
robustness.
