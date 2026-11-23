# UBP experiment report

Preliminary single-seed results (seed 1234, 300 epochs). Checkpoints were selected on internal validation patients from the training pool; supplied test partitions were evaluated afterwards.

## Aggregate test results

| Dataset | Test patients/cases | Patient mean Dice (bootstrap 95% CI) | Macro foreground HD95 (voxels) | Validation mean Dice |
|---|---:|---:|---:|---:|
| ACDC | 50 | 0.8914 (0.8834-0.8986) | 2.10 | 0.8822 |
| SYNAPSE | 12 | 0.7631 (0.7213-0.8039) | 35.32 | 0.5892 |

## ACDC per-class test metrics

| Class | Mean Dice | HD95 (voxels; valid / total) |
|---|---:|---:|
| Right ventricle (ID 1) | 0.8707 | 3.36 (100/100) |
| Myocardium (ID 2) | 0.8723 | 1.14 (100/100) |
| Left ventricle (ID 3) | 0.9312 | 1.80 (100/100) |

## SYNAPSE per-class test metrics

| Class | Mean Dice | HD95 (voxels; valid / total) |
|---|---:|---:|
| Aorta (ID 1) | 0.8289 | 22.16 (12/12) |
| Gallbladder (ID 2) | 0.5339 | 13.98 (10/12) |
| Left kidney (ID 3) | 0.8252 | 45.56 (12/12) |
| Right kidney (ID 4) | 0.7740 | 83.39 (12/12) |
| Liver (ID 5) | 0.9288 | 24.98 (12/12) |
| Pancreas (ID 6) | 0.5780 | 13.26 (12/12) |
| Spleen (ID 7) | 0.8576 | 59.12 (12/12) |
| Stomach (ID 8) | 0.7782 | 20.13 (12/12) |

## Compute measurements

| Dataset | Parameters | Partial FLOPs (G) | Batch-one median (ms) | Network slices/s | Full-volume slices/s | Peak VRAM (GiB) | Train time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACDC | 40,644,493 | 15.980 | 10.03 | 99.84 | 335.39 | 8.24 | 67.7 |
| SYNAPSE | 40,644,141 | 15.903 | 9.28 | 107.85 | 342.34 | 8.27 | 69.4 |

## Notes

- Bootstrap intervals resample patients/cases for this single trained seed; they do not capture variation across training seeds.
- HD95 is in voxel units. Physical image spacing was unavailable. Undefined empty/missing-class outcomes are retained in the raw JSON; macro HD95 averages defined class values.
- FLOPs count only profiler-supported convolution/matrix operators. Network latency is warmed-up batch-one; full-volume throughput includes preprocessing and restoration. Shared-GPU measurements are descriptive snapshots.
- Results are exploratory, not a claim of novelty, state-of-the-art performance, or clinical validity.

## Raw artifacts

Per-case metrics, profiles, status records, and epoch histories are in `raw/`. Run `python scripts/make_release_assets.py` to regenerate this report, the CSV/JSON summary, and aggregate figures.
