"""Evaluation entry point for one retrieval method on Charades-STA."""

from pathlib import Path
import csv
import hashlib
import json

import numpy as np

from data_utils import load_sta_annotations, load_video_lengths, project_paths
from model import DEFAULT_ALIGNMENT_MODEL, SigLIPEncoder, build_visual_feature_extractor, default_i3d_checkpoint
from retrieval import encode_video, retrieve_moment_from_features


PREDICTION_FIELDS = [
    "video_id",
    "query",
    "gt_start",
    "gt_end",
    "pred_start",
    "pred_end",
    "iou",
    "method",
    "sub_queries",
    "relations",
    "query_backend",
    "candidate_count",
    "candidate_score",
    "combination_count",
    "overlap_candidate_count",
    "selected_candidates",
    "final_candidate",
    "proposal_method",
    "proposal_k",
    "min_proposal_snippets",
    "overlap_required",
    "qc_lambda",
    "context_distance",
]


def safe_cache_name(text):
    """Create short stable filenames for model/video cache keys."""
    text = str(text).replace("\\", "/")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    stem = Path(text).stem if text else "item"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return f"{safe_stem}_{digest}"


def video_feature_cache_path(cache_dir, cache_key, video_path, num_snippets, snippet_frames):
    model_key = safe_cache_name(cache_key)
    video_key = safe_cache_name(video_path)
    return Path(cache_dir) / model_key / f"{video_key}_s{num_snippets}_c{snippet_frames}.npz"


def load_video_feature_cache(path):
    """Return cached video features, or None when the cache file is absent."""
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {
        "times": data["times"],
        "duration": float(data["duration"]),
        "visual_features": data["visual_features"],
        "alignment_features": data["alignment_features"],
        "metadata": json.loads(str(data["metadata"])),
    }


def save_video_feature_cache(path, video_features, metadata):
    """Persist encoded video features so repeated experiments can reuse them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=np.asarray(video_features["times"]),
        duration=np.asarray(video_features["duration"]),
        visual_features=np.asarray(video_features["visual_features"]),
        alignment_features=np.asarray(video_features["alignment_features"]),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def temporal_iou(pred_start, pred_end, gt_start, gt_end):
    """Intersection over union for one predicted and one ground-truth segment."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union


def summarize_ious(ious):
    """Compute the standard VMR metrics used in the report tables."""
    total = len(ious)
    return {
        "R@1_IoU_0.1": sum(iou >= 0.1 for iou in ious) / total,
        "R@1_IoU_0.3": sum(iou >= 0.3 for iou in ious) / total,
        "R@1_IoU_0.5": sum(iou >= 0.5 for iou in ious) / total,
        "R@1_IoU_0.7": sum(iou >= 0.7 for iou in ious) / total,
        "mIoU": sum(ious) / total,
    }


def write_predictions(path, rows):
    """Write detailed per-query predictions for later error analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def feature_cache_metadata(sample, alignment_model_name, i3d_checkpoint, i3d_num_classes, i3d_batch_size, num_frames, snippet_frames):
    """Store enough context to know how a cached feature file was produced."""
    return {
        "video_path": str(sample.video_path),
        "visual_backbone": "i3d",
        "i3d_checkpoint": str(i3d_checkpoint),
        "i3d_num_classes": i3d_num_classes,
        "i3d_batch_size": i3d_batch_size,
        "alignment_model_name": alignment_model_name,
        "num_snippets": num_frames,
        "snippet_frames": snippet_frames,
    }


def load_or_encode_video(sample, video_cache, cache_path, visual_extractor, alignment_encoder, num_frames, snippet_frames, metadata):
    """Reuse features across methods and persist them across runs when a cache path is provided."""
    video_key = str(sample.video_path)
    if video_key in video_cache:
        return video_cache[video_key]

    cached_features = load_video_feature_cache(cache_path) if cache_path is not None else None
    if cached_features is not None:
        video_cache[video_key] = cached_features
        return cached_features

    video_features = encode_video(
        sample.video_path,
        visual_extractor=visual_extractor,
        alignment_encoder=alignment_encoder,
        num_snippets=num_frames,
        snippet_frames=snippet_frames,
    )
    video_cache[video_key] = video_features
    if cache_path is not None:
        save_video_feature_cache(cache_path, video_features, metadata=metadata)
    return video_features


def prediction_row(sample, result, iou, method):
    """Flatten one retrieval result into a CSV-friendly row."""
    return {
        "video_id": sample.video_id,
        "query": sample.query,
        "gt_start": sample.start,
        "gt_end": sample.end,
        "pred_start": result["pred_start"],
        "pred_end": result["pred_end"],
        "iou": iou,
        "method": result.get("method", method),
        "sub_queries": json.dumps(result.get("sub_queries", [sample.query]), ensure_ascii=False),
        "relations": json.dumps(result.get("relations", []), ensure_ascii=False),
        "query_backend": result.get("query_backend", ""),
        "candidate_count": result.get("candidate_count", 0),
        "candidate_score": result.get("candidate_score", ""),
        "combination_count": result.get("combination_count", 0),
        "overlap_candidate_count": result.get("overlap_candidate_count", 0),
        "selected_candidates": json.dumps(result.get("selected_candidates", []), ensure_ascii=False),
        "final_candidate": json.dumps(result.get("final_candidate", {}), ensure_ascii=False),
        "proposal_method": result.get("proposal_method", ""),
        "proposal_k": result.get("proposal_k", ""),
        "min_proposal_snippets": result.get("min_proposal_snippets", ""),
        "overlap_required": result.get("overlap_required", ""),
        "qc_lambda": result.get("qc_lambda", ""),
        "context_distance": result.get("context_distance", ""),
    }


def run_eval(
    project_root,
    output_path,
    limit=None,
    num_frames=32,
    snippet_frames=16,
    smooth_kernel=5,
    threshold_ratio=0.75,
    alignment_model_name=DEFAULT_ALIGNMENT_MODEL,
    i3d_checkpoint=None,
    i3d_num_classes=400,
    i3d_batch_size=4,
    method="baseline",
    query_backend="rule",
    qc_lambda=0.5,
    context_distance=2,
    proposal_k=6,
    proposal_method="kmeans",
    require_overlap=True,
    min_proposal_snippets=0,
    alignment_encoder=None,
    visual_extractor=None,
    video_cache=None,
    feature_cache_dir=None,
):
    """Run one method on Charades-STA and write per-sample predictions."""
    paths = project_paths(project_root)
    project_root = Path(project_root)
    if i3d_checkpoint is None:
        i3d_checkpoint = default_i3d_checkpoint(project_root)

    lengths = load_video_lengths(paths["test_csv"])
    samples = load_sta_annotations(paths["sta_test"], paths["video_dir"], lengths, limit=limit)

    if alignment_encoder is None:
        alignment_encoder = SigLIPEncoder(model_name=alignment_model_name)
    if visual_extractor is None:
        visual_extractor = build_visual_feature_extractor(
            i3d_checkpoint=i3d_checkpoint,
            i3d_num_classes=i3d_num_classes,
            i3d_batch_size=i3d_batch_size,
        )

    if video_cache is None:
        video_cache = {}

    feature_cache_key = f"visual={visual_extractor.name}|align={alignment_encoder.name}"
    rows = []
    ious = []

    for index, sample in enumerate(samples, start=1):
        # Video encoding is the expensive part. Cache it in memory across
        # methods and optionally on disk across script runs.
        cache_path = None
        if feature_cache_dir is not None:
            cache_path = video_feature_cache_path(
                cache_dir=feature_cache_dir,
                cache_key=feature_cache_key,
                video_path=sample.video_path,
                num_snippets=num_frames,
                snippet_frames=snippet_frames,
            )

        video_features = load_or_encode_video(
            sample=sample,
            video_cache=video_cache,
            cache_path=cache_path,
            visual_extractor=visual_extractor,
            alignment_encoder=alignment_encoder,
            num_frames=num_frames,
            snippet_frames=snippet_frames,
            metadata=feature_cache_metadata(
                sample=sample,
                alignment_model_name=alignment_model_name,
                i3d_checkpoint=i3d_checkpoint,
                i3d_num_classes=i3d_num_classes,
                i3d_batch_size=i3d_batch_size,
                num_frames=num_frames,
                snippet_frames=snippet_frames,
            ),
        )

        result = retrieve_moment_from_features(
            video_features=video_features,
            query=sample.query,
            alignment_encoder=alignment_encoder,
            smooth_kernel=smooth_kernel,
            threshold_ratio=threshold_ratio,
            method=method,
            query_backend=query_backend,
            qc_lambda=qc_lambda,
            context_distance=context_distance,
            proposal_k=proposal_k,
            proposal_method=proposal_method,
            require_overlap=require_overlap,
            min_proposal_snippets=min_proposal_snippets,
        )
        iou = temporal_iou(
            result["pred_start"],
            result["pred_end"],
            sample.start,
            sample.end,
        )
        ious.append(iou)
        rows.append(prediction_row(sample, result, iou, method))

        if index % 50 == 0:
            print(f"Processed {index}/{len(samples)}")

    metrics = summarize_ious(ious)
    write_predictions(output_path, rows)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics, rows


def parse_limit(value):
    """Parse CLI/notebook limit values; 'full' and 'none' mean all samples."""
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"none", "full", "all"}:
        return None
    return int(value)

