# COMP5405 Project: Zero-Shot Video Moment Retrieval

This repository contains our COMP5405 project for **zero-shot video moment retrieval** on Charades-STA. Given an untrimmed video and a natural-language query, the system predicts the temporal segment where the described action happens.

The implementation is a course project rather than a full reproduction of a single paper. It keeps the paper-inspired ideas that are feasible in our setting - query decomposition, query-conditioned feature refinement, and bottom-up proposal generation - while replacing the original NLP parser and some feature components with lightweight local alternatives.

## Task

Input:

```text
video: Charades video
query: "the person opens the bag"
```

Output:

```text
predicted moment: start time - end time
```

The system is **zero-shot**: it does not train or fine-tune on the Charades-STA training split.

## Current Experiment Design

All reported full-data experiments use the Charades-STA test split with **3,720 query-video pairs**.

Shared settings:

| Setting | Value |
|---|---:|
| `num_frames` | 32 |
| `snippet_frames` | 16 |
| query parser | spaCy dependency parser |
| I3D checkpoint | `outputs/models/i3d/rgb_imagenet.pt` |
| proposal method | k-means |
| `proposal_k` | 6 |
| `qc_lambda` | 0.5 |
| `context_distance` | 2 |
| `min_proposal_snippets` | 0 |

`min_proposal_snippets=0` means the code uses the automatic minimum proposal length. This is a stabilization choice for our sampled-snippet setting, not a core claim from the original paper.

### Compared Methods

| Method | Meaning |
|---|---|
| `baseline` | Frozen VLM snippet-query similarity curve with smoothing, peak selection, and boundary expansion. |
| `query_decomp` | Baseline plus spaCy-based verb-centered query decomposition and score fusion. |
| `bu_pg` | Query decomposition plus bottom-up proposal generation over raw I3D snippet features. This is our **w/o QC-FR** ablation. |
| `full` | Query decomposition plus query-conditioned feature refinement (QC-FR) plus bottom-up proposal generation (BU-PG). |

### Backbone Experiments

The main ablation uses CLIP:

```text
openai/clip-vit-base-patch32
```

We also run SigLIP2 as a backbone replacement:

```text
google/siglip2-base-patch16-224
```

### Novel-location OOD-1

The Novel-location OOD-1 setting prepends 10 seconds from another Charades video before the target video, then shifts the ground-truth timestamps accordingly. This tests whether the model can still localize the queried moment when irrelevant video content appears before the original clip.

No concatenated videos are written to disk; the OOD split is simulated during sampling and evaluation.

## Methodology and Module Overview

The pipeline works at snippet level. Each video is divided into 32 temporal snippets, and each snippet is represented by both vision-language alignment features and I3D action features.

| Module | Purpose | Main file |
|---|---|---|
| Data loading | Reads Charades-STA annotations, video metadata, and local video paths. | `src/data_utils.py` |
| Video sampling | Uniformly samples snippet frames from each video, including the shifted sampling used for OOD-1. | `src/retrieval.py` |
| VLM alignment | Uses CLIP or SigLIP2 to encode query text and sampled frames, then produces a snippet-query score curve. | `src/model.py` |
| I3D extraction | Produces one action-aware feature vector for each sampled snippet. These features support proposal generation rather than direct text matching. | `src/model.py`, `src/i3d.py` |
| Baseline retrieval | Smooths the VLM score curve, selects the highest-scoring snippet, and expands the temporal boundary around it. | `src/retrieval.py` |
| Query decomposition | Uses spaCy dependency parsing to split complex queries into verb-centered sub-queries, then fuses their score curves. | `src/query_utils.py` |
| QC-FR | Refines I3D snippet features using query-score similarity and nearby temporal context. | `src/retrieval.py` |
| BU-PG | Clusters I3D features with k-means, converts clusters into temporal proposals, and selects the proposal with the best query alignment. | `src/retrieval.py` |
| Evaluation | Runs each method, writes prediction CSV files, and computes IoU-based metrics. | `src/eval.py`, `src/run_experiments.py` |

In short, CLIP/SigLIP2 decides which snippets are semantically close to the query, I3D provides motion-aware snippet features, BU-PG turns snippet features into candidate temporal segments, and QC-FR tests whether query-aware feature refinement improves those proposals.

## Full-data Results

The following metrics are from the full Charades-STA test split.

### Main CLIP Ablation

`outputs/experiments/main_ablation_clip_metrics.csv`

| Method | R@1 IoU 0.1 | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU |
|---|---:|---:|---:|---:|---:|
| baseline | 89.62 | 47.93 | 17.72 | 6.02 | 32.53 |
| query_decomp | 89.54 | 47.74 | 16.72 | 5.83 | 32.25 |
| bu_pg / w/o QC-FR | 64.09 | 51.61 | 38.63 | 15.46 | 34.35 |
| full | 62.04 | 50.89 | 38.25 | 15.43 | 33.71 |

Main observations:

- Query decomposition alone does not improve the CLIP baseline.
- BU-PG is the strongest CLIP variant in this run. Compared with the baseline, it improves R@1 IoU 0.5 by 20.91 points and R@1 IoU 0.7 by 9.44 points.
- The full model is close to BU-PG but slightly lower, suggesting that QC-FR is not clearly beneficial with the current CLIP + I3D feature setup.
- Proposal-based methods reduce R@1 IoU 0.1 because they predict more selective intervals. This hurts very loose overlap but helps stricter localization.

### Backbone Replacement

`outputs/experiments/backbone_siglip2_metrics.csv`

| Backbone | Method | R@1 IoU 0.1 | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU |
|---|---|---:|---:|---:|---:|---:|
| CLIP | baseline | 89.62 | 47.93 | 17.72 | 6.02 | 32.53 |
| CLIP | full | 62.04 | 50.89 | 38.25 | 15.43 | 33.71 |
| SigLIP2 | baseline | 78.98 | 57.23 | 31.77 | 11.32 | 36.07 |
| SigLIP2 | full | 66.80 | 56.34 | 43.41 | 18.01 | 37.47 |

SigLIP2 is stronger than CLIP for stricter localization. In the full model, SigLIP2 improves over CLIP by 5.16 points at R@1 IoU 0.5 and 3.76 points in mIoU.

### Novel-location OOD-1

`outputs/experiments/ood1_clip_metrics.csv`

| Backbone | Method | R@1 IoU 0.1 | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU |
|---|---|---:|---:|---:|---:|---:|
| CLIP | baseline | 85.59 | 33.71 | 9.19 | 3.01 | 26.19 |
| CLIP | full | 53.01 | 41.67 | 28.09 | 14.17 | 27.80 |

`outputs/experiments/ood1_siglip2_metrics.csv`

| Backbone | Method | R@1 IoU 0.1 | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU |
|---|---|---:|---:|---:|---:|---:|
| SigLIP2 | baseline | 76.32 | 47.98 | 24.57 | 9.44 | 32.06 |
| SigLIP2 | full | 62.10 | 49.73 | 34.62 | 17.31 | 33.30 |

OOD observations:

- The prepended irrelevant content hurts all methods, especially low-precision peak-based retrieval.
- The full model remains better than the baseline at stricter IoU thresholds for both CLIP and SigLIP2.
- SigLIP2 remains the stronger backbone under OOD-1.

## Interpretation

The current results support three main conclusions:

1. **Proposal generation is useful for stricter temporal localization.**  
   The baseline often overlaps the ground truth loosely, which explains its high R@1 IoU 0.1. BU-PG and the full method produce more precise intervals, so they trade off loose recall for much better R@1 IoU 0.5 and 0.7.

2. **QC-FR is not consistently positive in our current setup.**  
   On CLIP, `bu_pg` is slightly better than `full`. This means our I3D refinement is not adding a reliable gain beyond raw I3D proposal generation. We should present QC-FR as a tested module with mixed effect, not as a guaranteed improvement.

3. **The VLM backbone matters.**  
   SigLIP2 improves both the baseline and full model at stricter IoU thresholds and under OOD-1. This is a clean comparison because the rest of the pipeline settings are kept the same.

## Repository Layout

```text
project/
  src/
    data_utils.py          # Charades-STA metadata and annotation loading
    eval.py                # single-method evaluation logic
    i3d.py                 # Inception-I3D network definition
    model.py               # CLIP/SigLIP2 alignment and I3D wrappers
    query_utils.py         # spaCy-based query decomposition
    retrieval.py           # retrieval, QC-FR, and BU-PG logic
    run_experiments.py     # multi-method experiment entry point
  notebooks/
    01_data_and_sampling_check.ipynb
    02_comparison_experiments.ipynb
    03_report_visualizations.ipynb
  requirements.txt
  README.md
```

## Data Setup

The dataset and model weights are not included in this repository.

Expected local layout:

```text
Charades/
  Charades/
    Charades_v1_test.csv
    charades_sta_test.txt
  Charades_v1_480/
    Charades_v1_480/
      <video_id>.mp4
outputs/
  models/
    i3d/
      rgb_imagenet.pt
```

External resources:

- Charades dataset: <https://prior.allenai.org/projects/charades>
- Charades-STA annotations: <https://github.com/jiyanggao/TALL>
- Charades-STA test-set mirror used for setup reference: <https://huggingface.co/datasets/jwnt4/charades-sta-test>

## Environment

Install dependencies:

```powershell
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

CUDA-enabled PyTorch is strongly recommended. CPU execution works for small checks but is slow for full-data runs.

## Running Experiments

Main CLIP ablation:

```powershell
python .\src\run_experiments.py --limit full --num-frames 32 --snippet-frames 16 --methods baseline query_decomp bu_pg full --alignment-model-name openai/clip-vit-base-patch32 --query-backend spacy --i3d-batch-size 8 --alignment-image-batch-size 64 --min-proposal-snippets 0 --experiment-name main_ablation_clip
```

SigLIP2 backbone replacement:

```powershell
python .\src\run_experiments.py --limit full --num-frames 32 --snippet-frames 16 --methods baseline full --alignment-model-name google/siglip2-base-patch16-224 --query-backend spacy --i3d-batch-size 8 --alignment-image-batch-size 64 --min-proposal-snippets 0 --experiment-name backbone_siglip2
```

Novel-location OOD-1 with CLIP:

```powershell
python .\src\run_experiments.py --limit full --num-frames 32 --snippet-frames 16 --methods baseline full --alignment-model-name openai/clip-vit-base-patch32 --query-backend spacy --i3d-batch-size 8 --alignment-image-batch-size 64 --min-proposal-snippets 0 --novel-location-prefix-seconds 10 --novel-location-seed 0 --experiment-name ood1_clip
```

Novel-location OOD-1 with SigLIP2:

```powershell
python .\src\run_experiments.py --limit full --num-frames 32 --snippet-frames 16 --methods baseline full --alignment-model-name google/siglip2-base-patch16-224 --query-backend spacy --i3d-batch-size 8 --alignment-image-batch-size 64 --min-proposal-snippets 0 --novel-location-prefix-seconds 10 --novel-location-seed 0 --experiment-name ood1_siglip2
```

Generated outputs:

```text
outputs/experiments/<experiment_name>_metrics.csv
outputs/experiments/<experiment_name>_metrics.json
outputs/predictions/<experiment_name>/<method>.csv
outputs/report_figures/final_report/*.png
```

The final metrics, predictions, and report figures are included in Git so the reported results can be inspected without rerunning the full experiment. Large local artifacts such as `outputs/cache/` and `outputs/models/` remain ignored.

## Notebooks

Use notebooks in this order:

1. `01_data_and_sampling_check.ipynb`  
   Check local paths, annotation loading, video availability, and frame sampling.

2. `02_comparison_experiments.ipynb`  
   Run the report experiment matrix: CLIP ablation, SigLIP2 backbone replacement, and OOD-1 checks.

3. `03_report_visualizations.ipynb`  
   Generate report tables and figures from saved metrics, predictions, and original videos.

Notebook outputs are intentionally cleared before GitHub upload.

## Report Figures

`03_report_visualizations.ipynb` generates:

- `main_ablation_metrics.png`
- `backbone_replacement.png`
- `ood1_backbone_comparison.png`
- `main_iou_delta_vs_baseline.png`
- `sample_walkthrough.png`
- `video_frame_walkthrough.png`
