# COMP5405 Project: Zero-Shot Video Moment Retrieval

This repository contains our COMP5405 project implementation for **zero-shot video moment retrieval** on Charades-STA. Given a video and a natural-language query, the system predicts the temporal segment where the described action happens.

The project is a course implementation, not a full reproduction of one specific paper. It combines pretrained video and vision-language features with lightweight retrieval, query decomposition, feature refinement, and proposal generation.

## What The Project Does

Input:

```text
video: Charades video
query: "the person opens the bag"
```

Output:

```text
predicted moment: start time - end time
```

The pipeline supports five methods:

- `baseline`: SigLIP2 snippet-query score curve with smoothing and peak expansion.
- `query_decomp`: rule-based query decomposition followed by score fusion.
- `qc_fr`: query-conditioned feature refinement over I3D snippet features.
- `bu_pg`: k-means bottom-up proposal generation.
- `full`: query decomposition + QC-FR + BU-PG.

The project is **zero-shot**: it does not train on the Charades-STA training split.

## Repository Layout

```text
project/
  src/
    data_utils.py          # Charades-STA metadata and annotation loading
    eval.py                # single-method evaluation logic
    i3d.py                 # self-contained Inception-I3D network definition
    model.py               # SigLIP2 and I3D wrappers
    query_utils.py         # rule-based query decomposition
    retrieval.py           # core retrieval, QC-FR, and BU-PG logic
    run_experiments.py     # multi-method comparison entry point
  notebooks/
    01_data_and_sampling_check.ipynb
    03_comparison_experiments.ipynb
    04_report_visualizations.ipynb
  requirements.txt
  README.md
```

## Data Sources

The dataset and model weights are **not included** in this GitHub repository.

We use the following external resources:

- Charades dataset official page: <https://prior.allenai.org/projects/charades>
- Charades videos and metadata: use the official Charades downloads, especially the 480p video package.
- Charades-STA annotations: introduced by TALL, with the original repository at <https://github.com/jiyanggao/TALL>
- Convenient Charades-STA test-set mirror used for setup reference: <https://huggingface.co/datasets/jwnt4/charades-sta-test>

The AllenAI Charades page describes the original Charades dataset as 9,848 indoor daily-activity videos with temporal action annotations, object labels, and text descriptions. Charades-STA adds sentence-level temporal annotations on top of Charades for temporal language grounding.

## Local Data Setup

After downloading the data locally, place files in this layout:

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

Expected local files:

- `Charades/Charades/charades_sta_test.txt`
- `Charades/Charades/Charades_v1_test.csv`
- `Charades/Charades_v1_480/Charades_v1_480/*.mp4`
- `outputs/models/i3d/rgb_imagenet.pt`

The I3D checkpoint should be a compatible RGB Inception-I3D checkpoint. The repository does not include this model file because it is large.

## Environment

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

CUDA-enabled PyTorch is strongly recommended for the full experiment. CPU execution works for small checks but is slow.

If CUDA PyTorch is needed, install the correct PyTorch build from the official PyTorch installation page before running the full experiment.

## Methodology Summary

The main retrieval logic is implemented in `src/retrieval.py`.

1. **Video sampling**  
   Each video is uniformly divided into snippets. Each snippet contains several sampled frames.

2. **Visual feature extraction**  
   I3D extracts one action-aware visual feature per snippet.

3. **Text-video alignment**  
   SigLIP2 encodes query text and sampled frames. Frame embeddings are averaged into snippet embeddings, then compared with the query embedding.

4. **Baseline localization**  
   The baseline builds a snippet-level similarity curve, smooths it, selects the peak, and expands temporal boundaries around the peak.

5. **Query decomposition**  
   A rule-based parser splits complex queries into simpler sub-queries and identifies simple temporal relations.

6. **Query-conditioned feature refinement**  
   QC-FR refines I3D features using query-score similarity and nearby temporal context.

7. **Bottom-up proposal generation**  
   BU-PG clusters snippet features with k-means, converts clusters into candidate temporal proposals, and selects the best proposal using query similarity.

8. **Full method**  
   The full pipeline combines query decomposition, QC-FR, and BU-PG.

## Run Experiments

Small comparison for report development:

```powershell
python .\src\run_experiments.py --limit 100 --num-frames 16 --snippet-frames 16 --methods baseline query_decomp qc_fr bu_pg full --i3d-batch-size 1 --experiment-name compare_l100_i3d_siglip2_minprop_auto
```

Full test-set comparison:

```powershell
python .\src\run_experiments.py --limit full --num-frames 32 --snippet-frames 16 --methods baseline query_decomp qc_fr bu_pg full --i3d-batch-size 1 --experiment-name compare_full_i3d_siglip2_minprop_auto
```

Generated local outputs:

- `outputs/experiments/<experiment_name>_metrics.csv`
- `outputs/experiments/<experiment_name>_metrics.json`
- `outputs/predictions/<experiment_name>/<method>.csv`
- `outputs/report_figures/<experiment_name>/*.png`

These output folders are ignored by Git and should not be uploaded.

## Notebooks

Use notebooks in this order:

1. `01_data_and_sampling_check.ipynb`  
   Check local paths, annotation loading, video availability, and sampled frames.

2. `02_comparison_experiments.ipynb`  
   Run small and full comparison experiments.

3. `03_report_visualizations.ipynb`  
   Generate report-ready figures from saved metrics, predictions, and original videos.

Notebook outputs are intentionally cleared before GitHub upload.

## Report Figures

`04_report_visualizations.ipynb` generates:

- `paper_metrics_comparison.png`
- `iou_delta_vs_baseline.png`
- `length_alignment.png`
- `qualitative_timelines.png`
- `sample_walkthrough.png`
- `video_frame_walkthrough.png`

The frame walkthrough figure directly samples frames from the original `.mp4` video, so it is useful for qualitative report analysis.

## Current Small-Set Result

Experiment:

```text
compare_l100_i3d_siglip2_minprop_auto
```

| Method | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU |
|---|---:|---:|---:|
| baseline | 0.29 | 0.08 | 0.3716 |
| query_decomp | 0.32 | 0.08 | 0.3718 |
| qc_fr | 0.48 | 0.24 | 0.4036 |
| bu_pg | 0.48 | 0.28 | 0.4142 |
| full | 0.49 | 0.26 | 0.4064 |

These results are from a 100-query development run and are mainly used for report preparation.