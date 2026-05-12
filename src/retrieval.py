"""Retrieval pipeline for zero-shot Charades-STA moment localization.

The code mirrors the report structure:
1. sample video snippets;
2. score snippet-query alignment with a frozen CLIP/SigLIP-style encoder;
3. optionally refine I3D features with QC-FR;
4. optionally generate BU-PG proposals.
"""

from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from model import cosine_scores
from query_utils import parse_query


EPS = 1e-8
DEFAULT_QC_LAMBDA = 0.5
DEFAULT_CONTEXT_DISTANCE = 2
DEFAULT_PROPOSAL_K = 6
DEFAULT_PROPOSAL_METHOD = "kmeans"
DEFAULT_MIN_PROPOSAL_SNIPPETS = 0
DEFAULT_AUTO_MIN_PROPOSAL_RATIO = 0.25
SUPPORTED_METHODS = ["baseline", "query_decomp", "qc_fr", "bu_pg", "full"]
PROPOSAL_METHODS = ["kmeans"]


def open_video_stream(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Could not open a valid video stream from {video_path}")
    duration = frame_count / fps
    return cap, fps, frame_count, duration


def read_frame_at_time(cap, fps, frame_count, time):
    frame_index = min(max(int(time * fps), 0), frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def sample_video_snippets(
    video_path,
    num_snippets=32,
    snippet_frames=16,
    prefix_video_path=None,
    prefix_seconds=0.0,
):
    """Uniformly sample snippets and return center timestamps.

    When `prefix_video_path` and `prefix_seconds` are provided, the sampler
    behaves as if the first `prefix_seconds` seconds of a different video were
    prepended before the target video. This supports Novel-location OOD
    evaluation without writing temporary concatenated videos.
    """
    target_cap, target_fps, target_frame_count, target_duration = open_video_stream(video_path)
    prefix_seconds = max(0.0, float(prefix_seconds or 0.0))
    prefix_cap = None
    prefix_fps = None
    prefix_frame_count = None
    prefix_duration = None
    if prefix_seconds > 0:
        if prefix_video_path is None:
            target_cap.release()
            raise ValueError("prefix_video_path is required when prefix_seconds > 0")
        prefix_cap, prefix_fps, prefix_frame_count, prefix_duration = open_video_stream(prefix_video_path)

    duration = target_duration + prefix_seconds

    snippets = []
    centers = []
    try:
        for snippet_index in range(num_snippets):
            # Each snippet covers an equal temporal bin. The model later treats
            # the bin center as the snippet timestamp.
            start = duration * snippet_index / num_snippets
            end = duration * (snippet_index + 1) / num_snippets
            step = (end - start) / max(snippet_frames, 1)
            times = start + step * (np.arange(snippet_frames) + 0.5)
            frames = []
            for time in times:
                if prefix_cap is not None and time < prefix_seconds:
                    source_time = min(time, max(prefix_duration - EPS, 0.0))
                    frame = read_frame_at_time(prefix_cap, prefix_fps, prefix_frame_count, source_time)
                else:
                    source_time = min(max(time - prefix_seconds, 0.0), max(target_duration - EPS, 0.0))
                    frame = read_frame_at_time(target_cap, target_fps, target_frame_count, source_time)
                if frame is not None:
                    frames.append(frame)

            if not frames and snippets:
                frames = snippets[-1]
            if not frames:
                raise RuntimeError(f"Could not decode frames from {video_path}")

            snippets.append(frames)
            centers.append(0.5 * (start + end))
    finally:
        target_cap.release()
        if prefix_cap is not None:
            prefix_cap.release()

    return np.asarray(centers), snippets, duration


def smooth_scores(scores, kernel_size=5):
    scores = np.asarray(scores, dtype=float)
    if kernel_size <= 1:
        return scores
    kernel_size = min(int(kernel_size), len(scores))
    if kernel_size <= 1:
        return scores
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(scores, kernel, mode="same")


def minmax_normalize(scores):
    scores = np.asarray(scores, dtype=float)
    low = float(scores.min())
    high = float(scores.max())
    if high - low < EPS:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def l2_normalize(values):
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def compute_score_matrix(alignment_features, text_features, smooth_kernel=1):
    """Compute f^c(s, q): snippet-query similarity from frozen alignment features."""
    alignment_features = l2_normalize(alignment_features)
    text_features = l2_normalize(text_features)
    if text_features.ndim == 1:
        text_features = text_features[None, :]

    score_matrix = alignment_features @ text_features.T
    if smooth_kernel > 1:
        score_matrix = np.asarray(
            [
                smooth_scores(score_matrix[:, index], kernel_size=smooth_kernel)
                for index in range(score_matrix.shape[1])
            ]
        ).T
    return score_matrix


def query_conditioned_feature_refinement(
    visual_features,
    score_matrix,
    context_distance=DEFAULT_CONTEXT_DISTANCE,
    qc_lambda=DEFAULT_QC_LAMBDA,
):
    """QC-FR: refine I3D features using query-conditioned context weights."""
    features = l2_normalize(visual_features)
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]

    length, dim = features.shape
    query_count = scores.shape[1]
    refined = np.zeros((query_count, length, dim), dtype=float)
    snippet_indices = np.arange(length)
    context_distance = max(0, int(context_distance))

    for query_index in range(query_count):
        query_scores = scores[:, query_index]
        for snippet_index in range(length):
            # Similar score to the current snippet means "likely same moment";
            # dissimilar score means "likely boundary or different action".
            differences = (query_scores[snippet_index] - query_scores) ** 2
            max_difference = float(differences.max())
            if max_difference < EPS:
                weights = np.ones(length, dtype=float)
            else:
                weights = 1.0 - differences / max_difference
                weights = np.clip(weights, 0.0, 1.0)

            mask = np.abs(snippet_indices - snippet_index) <= context_distance
            context_weights = weights * mask.astype(float)
            context_feature = np.sum(features * context_weights[:, None], axis=0)
            # Add weighted local context to the original visual feature. This is
            # the feature sequence used by proposal generation.
            refined[query_index, snippet_index] = features[snippet_index] + qc_lambda * context_feature

    return l2_normalize(refined)


def fuse_subquery_scores(score_matrix, parsed_query, times, duration):
    """Fuse simple-query score curves for the non-proposal query decomposition baseline."""
    if score_matrix.shape[1] == 1:
        return score_matrix[:, 0]

    normalized = np.asarray(
        [minmax_normalize(score_matrix[:, index]) for index in range(score_matrix.shape[1])]
    ).T
    mean_curve = normalized.mean(axis=1)
    max_curve = normalized.max(axis=1)
    relation_types = {relation.relation for relation in parsed_query.relations}

    if "sequential" not in relation_types:
        if "parallel" in relation_types:
            # Parallel actions benefit from both mean evidence and any strong
            # sub-query hit.
            return 0.65 * mean_curve + 0.35 * max_curve
        return mean_curve

    # For sequential clauses, softly bias earlier clauses toward earlier video
    # timestamps and later clauses toward later timestamps.
    relative_time = np.asarray(times, dtype=float) / max(float(duration), EPS)
    ordered_curve = np.zeros_like(mean_curve)
    count = normalized.shape[1]
    sigma = max(0.16, 0.55 / count)

    for index in range(count):
        center = (index + 0.5) / count
        temporal_weight = np.exp(-0.5 * ((relative_time - center) / sigma) ** 2)
        ordered_curve += normalized[:, index] * temporal_weight

    ordered_curve = minmax_normalize(ordered_curve / count)
    return 0.45 * mean_curve + 0.25 * max_curve + 0.30 * ordered_curve


def predict_segment_from_scores(scores, times, duration, threshold_ratio=0.75):
    """Strong baseline: choose the peak and expand while scores stay above threshold."""
    scores = np.asarray(scores, dtype=float)
    peak = int(np.argmax(scores))
    low = float(scores.min())
    high = float(scores.max())
    threshold = low + threshold_ratio * (high - low)

    left = peak
    while left > 0 and scores[left - 1] >= threshold:
        left -= 1

    right = peak
    while right < len(scores) - 1 and scores[right + 1] >= threshold:
        right += 1

    step = duration / len(scores)
    start = max(0.0, float(times[left] - step / 2))
    end = min(duration, float(times[right] + step / 2))
    return start, end


def candidate_from_indices(scores, times, duration, start_index, end_index, proposal_method, label=None):
    start_index = int(max(0, start_index))
    end_index = int(min(len(scores) - 1, end_index))
    if end_index < start_index:
        start_index, end_index = end_index, start_index

    step = duration / len(scores)
    start = max(0.0, float(times[start_index] - step / 2))
    end = min(duration, float(times[end_index] + step / 2))
    local_scores = np.asarray(scores[start_index : end_index + 1], dtype=float)
    return {
        "start": start,
        "end": end,
        "center": 0.5 * (start + end),
        "score": float(local_scores.mean()),
        "start_index": start_index,
        "end_index": end_index,
        "label": None if label is None else int(label),
        "proposal_method": proposal_method,
    }


def resolve_min_proposal_snippets(min_proposal_snippets, total_snippets):
    if min_proposal_snippets is None:
        return 1

    value = int(min_proposal_snippets)
    if value <= 0:
        # Charades-STA moments average roughly a quarter of video duration.
        value = int(np.ceil(total_snippets * DEFAULT_AUTO_MIN_PROPOSAL_RATIO))

    return max(1, min(value, total_snippets))


def expand_candidate_to_min_snippets(candidate, scores, times, duration, min_proposal_snippets):
    start_index = int(candidate["start_index"])
    end_index = int(candidate["end_index"])
    original_indices = (start_index, end_index)

    while end_index - start_index + 1 < min_proposal_snippets:
        can_expand_left = start_index > 0
        can_expand_right = end_index < len(scores) - 1
        if not can_expand_left and not can_expand_right:
            break

        left_score = scores[start_index - 1] if can_expand_left else -float("inf")
        right_score = scores[end_index + 1] if can_expand_right else -float("inf")
        # Grow toward the neighboring snippet that already looks more relevant
        # to the query, instead of expanding symmetrically.
        if right_score >= left_score:
            end_index += 1
        else:
            start_index -= 1

    expanded = candidate_from_indices(
        scores=scores,
        times=times,
        duration=duration,
        start_index=start_index,
        end_index=end_index,
        proposal_method=candidate["proposal_method"],
        label=candidate.get("label"),
    )
    expanded["min_proposal_snippets"] = int(min_proposal_snippets)
    if original_indices != (start_index, end_index):
        expanded["expanded_from"] = list(original_indices)
    return expanded


def deduplicate_candidates(candidates):
    best_by_span = {}
    for candidate in candidates:
        key = (candidate["start_index"], candidate["end_index"])
        if key not in best_by_span or candidate["score"] > best_by_span[key]["score"]:
            best_by_span[key] = candidate
    result = list(best_by_span.values())
    result.sort(key=lambda candidate: candidate["score"], reverse=True)
    return result


def kmeans_labels(features, k, max_iter=50):
    features = l2_normalize(features)
    length = len(features)
    k = max(1, min(int(k), length))
    if length == 1:
        return np.zeros(1, dtype=int)

    init_indices = np.linspace(0, length - 1, k, dtype=int)
    centers = features[init_indices].copy()
    previous_labels = None

    for _ in range(max_iter):
        distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(distances, axis=1)
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            break
        previous_labels = labels.copy()

        nearest_distances = distances[np.arange(length), labels]
        for cluster_index in range(k):
            members = features[labels == cluster_index]
            if len(members) > 0:
                centers[cluster_index] = members.mean(axis=0)
            else:
                centers[cluster_index] = features[int(np.argmax(nearest_distances))]
        centers = l2_normalize(centers)

    distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    return np.argmin(distances, axis=1)


def kmeans_proposals(features, scores, times, duration, proposal_k):
    """Cluster snippet features; each cluster's temporal extent becomes one proposal."""
    features = l2_normalize(features)
    labels = kmeans_labels(features, proposal_k)
    candidates = []
    for label in sorted(int(label) for label in np.unique(labels)):
        indices = np.where(labels == label)[0]
        candidates.append(
            candidate_from_indices(
                scores=scores,
                times=times,
                duration=duration,
                start_index=int(indices.min()),
                end_index=int(indices.max()),
                proposal_method="kmeans",
                label=label,
            )
        )
    return deduplicate_candidates(candidates)[: max(1, int(proposal_k))]


def generate_proposals_from_features(
    features,
    scores,
    times,
    duration,
    proposal_k=DEFAULT_PROPOSAL_K,
    proposal_method=DEFAULT_PROPOSAL_METHOD,
    min_proposal_snippets=DEFAULT_MIN_PROPOSAL_SNIPPETS,
):
    proposal_method = proposal_method.lower()
    if proposal_method != "kmeans":
        raise ValueError(f"Unsupported proposal method: {proposal_method}")

    # BU-PG uses the temporal span of each I3D feature cluster as a proposal.
    candidates = kmeans_proposals(features, scores, times, duration, proposal_k)

    min_snippets = resolve_min_proposal_snippets(
        min_proposal_snippets=min_proposal_snippets,
        total_snippets=len(scores),
    )
    candidates = [
        expand_candidate_to_min_snippets(
            candidate=candidate,
            scores=scores,
            times=times,
            duration=duration,
            min_proposal_snippets=min_snippets,
        )
        for candidate in candidates
    ]
    candidates = deduplicate_candidates(candidates)
    if not candidates:
        candidates = [
            candidate_from_indices(
                scores=scores,
                times=times,
                duration=duration,
                start_index=0,
                end_index=len(scores) - 1,
                proposal_method=proposal_method,
                label=0,
            )
        ]
        candidates = [
            expand_candidate_to_min_snippets(
                candidate=candidates[0],
                scores=scores,
                times=times,
                duration=duration,
                min_proposal_snippets=min_snippets,
            )
        ]
    return candidates[: max(1, int(proposal_k))]


def has_common_overlap(combination):
    intersection_start = max(candidate["start"] for candidate in combination)
    intersection_end = min(candidate["end"] for candidate in combination)
    return intersection_end > intersection_start


def union_candidate(combination, score, proposal_method, overlap_required, accepted_by_overlap):
    start = min(candidate["start"] for candidate in combination)
    end = max(candidate["end"] for candidate in combination)
    return {
        "start": start,
        "end": end,
        "center": 0.5 * (start + end),
        "score": float(score),
        "proposal_method": proposal_method,
        "overlap_required": bool(overlap_required),
        "accepted_by_overlap": bool(accepted_by_overlap),
    }


def bottom_up_proposal_generation(
    feature_sets,
    raw_score_matrix,
    times,
    duration,
    proposal_k=DEFAULT_PROPOSAL_K,
    proposal_method=DEFAULT_PROPOSAL_METHOD,
    require_overlap=True,
    min_proposal_snippets=DEFAULT_MIN_PROPOSAL_SNIPPETS,
):
    """BU-PG: enumerate simple-query proposals and score their overlapping unions."""
    raw_score_matrix = np.asarray(raw_score_matrix, dtype=float)
    if raw_score_matrix.ndim == 1:
        raw_score_matrix = raw_score_matrix[:, None]

    feature_sets = np.asarray(feature_sets, dtype=float)
    if feature_sets.ndim == 2:
        feature_sets = np.repeat(feature_sets[None, :, :], raw_score_matrix.shape[1], axis=0)

    candidate_lists = [
        generate_proposals_from_features(
            features=feature_sets[query_index],
            scores=raw_score_matrix[:, query_index],
            times=times,
            duration=duration,
            proposal_k=proposal_k,
            proposal_method=proposal_method,
            min_proposal_snippets=min_proposal_snippets,
        )
        for query_index in range(raw_score_matrix.shape[1])
    ]

    resolved_min_proposal_snippets = resolve_min_proposal_snippets(
        min_proposal_snippets=min_proposal_snippets,
        total_snippets=len(times),
    )

    if len(candidate_lists) == 1:
        # Single-query case: BU-PG degenerates to selecting the best proposal.
        best = candidate_lists[0][0]
        return best["start"], best["end"], {
            "candidate_score": best["score"],
            "candidate_count": len(candidate_lists[0]),
            "combination_count": len(candidate_lists[0]),
            "overlap_candidate_count": len(candidate_lists[0]),
            "selected_candidates": [best],
            "proposal_method": proposal_method,
            "proposal_k": proposal_k,
            "min_proposal_snippets": resolved_min_proposal_snippets,
            "overlap_required": require_overlap,
        }

    best_final = None
    best_combination = None
    best_score = -float("inf")
    combination_count = 0
    overlap_candidate_count = 0
    fallback = None
    fallback_combination = None
    fallback_score = -float("inf")

    for combination in product(*candidate_lists):
        # Multi-query case: enumerate one proposal per simple query, require
        # temporal overlap by default, then union accepted proposals.
        combination_count += 1
        score = float(np.mean([candidate["score"] for candidate in combination]))
        accepted_by_overlap = has_common_overlap(combination)

        if score > fallback_score:
            fallback_score = score
            fallback_combination = combination
            fallback = union_candidate(
                combination=combination,
                score=score,
                proposal_method=proposal_method,
                overlap_required=False,
                accepted_by_overlap=accepted_by_overlap,
            )

        if require_overlap and not accepted_by_overlap:
            continue

        overlap_candidate_count += 1
        candidate = union_candidate(
            combination=combination,
            score=score,
            proposal_method=proposal_method,
            overlap_required=require_overlap,
            accepted_by_overlap=accepted_by_overlap,
        )
        if score > best_score:
            best_score = score
            best_final = candidate
            best_combination = combination

    if best_final is None:
        best_final = fallback
        best_combination = fallback_combination
        best_score = fallback_score

    return best_final["start"], best_final["end"], {
        "candidate_score": best_score,
        "candidate_count": sum(len(candidates) for candidates in candidate_lists),
        "combination_count": combination_count,
        "overlap_candidate_count": overlap_candidate_count,
        "selected_candidates": list(best_combination),
        "final_candidate": best_final,
        "proposal_method": proposal_method,
        "proposal_k": proposal_k,
        "min_proposal_snippets": resolved_min_proposal_snippets,
        "overlap_required": require_overlap,
    }


def encode_video(
    video_path,
    visual_extractor,
    alignment_encoder,
    num_snippets=32,
    snippet_frames=16,
    prefix_video_path=None,
    prefix_seconds=0.0,
):
    """Encode one video once into I3D visual features and alignment features."""
    times, snippets, duration = sample_video_snippets(
        Path(video_path),
        num_snippets=num_snippets,
        snippet_frames=snippet_frames,
        prefix_video_path=prefix_video_path,
        prefix_seconds=prefix_seconds,
    )
    visual_features = visual_extractor.extract(snippets=snippets)
    alignment_features = alignment_encoder.encode_snippets(snippets)
    return {
        "times": times,
        "duration": duration,
        "visual_features": visual_features,
        "alignment_features": alignment_features,
    }


def retrieve_baseline_from_features(video_features, query, alignment_encoder, smooth_kernel=5, threshold_ratio=0.75):
    """Baseline: no I3D proposal logic, only SigLIP2 score curve localization."""
    text_feature = alignment_encoder.encode_text([query])[0]
    scores = cosine_scores(video_features["alignment_features"], text_feature)
    scores = smooth_scores(scores, kernel_size=smooth_kernel)
    start, end = predict_segment_from_scores(
        scores=scores,
        times=video_features["times"],
        duration=video_features["duration"],
        threshold_ratio=threshold_ratio,
    )
    return {
        "pred_start": start,
        "pred_end": end,
        "scores": scores,
        "method": "baseline",
        "sub_queries": [query],
        "relations": [],
    }


def retrieve_query_decomp_from_features(
    video_features,
    query,
    alignment_encoder,
    smooth_kernel=5,
    threshold_ratio=0.75,
    query_backend="spacy",
):
    """Query-decomposition baseline without QC-FR or BU-PG."""
    parsed_query = parse_query(query, backend=query_backend)
    text_features = alignment_encoder.encode_text(parsed_query.sub_queries)
    score_matrix = compute_score_matrix(
        alignment_features=video_features["alignment_features"],
        text_features=text_features,
        smooth_kernel=smooth_kernel,
    )
    scores = fuse_subquery_scores(
        score_matrix=score_matrix,
        parsed_query=parsed_query,
        times=video_features["times"],
        duration=video_features["duration"],
    )
    start, end = predict_segment_from_scores(
        scores=scores,
        times=video_features["times"],
        duration=video_features["duration"],
        threshold_ratio=threshold_ratio,
    )
    return {
        "pred_start": start,
        "pred_end": end,
        "scores": scores,
        "subquery_scores": score_matrix,
        "method": "query_decomp",
        "sub_queries": parsed_query.sub_queries,
        "relations": parsed_query.as_dict()["relations"],
        "query_backend": parsed_query.backend,
        "candidate_count": 0,
    }


def retrieve_qc_fr_from_features(
    video_features,
    query,
    alignment_encoder,
    qc_lambda=DEFAULT_QC_LAMBDA,
    context_distance=DEFAULT_CONTEXT_DISTANCE,
    proposal_k=DEFAULT_PROPOSAL_K,
    proposal_method=DEFAULT_PROPOSAL_METHOD,
    min_proposal_snippets=DEFAULT_MIN_PROPOSAL_SNIPPETS,
):
    """QC-FR ablation: refine I3D features using the raw query only."""
    text_feature = alignment_encoder.encode_text([query])
    raw_score_matrix = compute_score_matrix(
        alignment_features=video_features["alignment_features"],
        text_features=text_feature,
        smooth_kernel=1,
    )
    refined_features = query_conditioned_feature_refinement(
        visual_features=video_features["visual_features"],
        score_matrix=raw_score_matrix,
        context_distance=context_distance,
        qc_lambda=qc_lambda,
    )
    start, end, proposal_info = bottom_up_proposal_generation(
        feature_sets=refined_features,
        raw_score_matrix=raw_score_matrix,
        times=video_features["times"],
        duration=video_features["duration"],
        proposal_k=proposal_k,
        proposal_method=proposal_method,
        require_overlap=True,
        min_proposal_snippets=min_proposal_snippets,
    )
    return {
        "pred_start": start,
        "pred_end": end,
        "scores": raw_score_matrix[:, 0],
        "raw_scores": raw_score_matrix,
        "method": "qc_fr",
        "sub_queries": [query],
        "relations": [],
        "query_backend": "",
        "candidate_count": proposal_info["candidate_count"],
        "candidate_score": proposal_info["candidate_score"],
        "selected_candidates": proposal_info["selected_candidates"],
        "final_candidate": proposal_info.get("final_candidate", {}),
        "combination_count": proposal_info.get("combination_count", 0),
        "overlap_candidate_count": proposal_info.get("overlap_candidate_count", 0),
        "proposal_method": proposal_method,
        "proposal_k": proposal_k,
        "min_proposal_snippets": proposal_info.get("min_proposal_snippets", ""),
        "overlap_required": True,
        "qc_lambda": qc_lambda,
        "context_distance": context_distance,
    }


def retrieve_bu_pg_from_features(
    video_features,
    query,
    alignment_encoder,
    query_backend="spacy",
    use_qc_fr=False,
    qc_lambda=DEFAULT_QC_LAMBDA,
    context_distance=DEFAULT_CONTEXT_DISTANCE,
    proposal_k=DEFAULT_PROPOSAL_K,
    proposal_method=DEFAULT_PROPOSAL_METHOD,
    require_overlap=True,
    min_proposal_snippets=DEFAULT_MIN_PROPOSAL_SNIPPETS,
):
    """BU-PG method; when use_qc_fr=True this becomes the full pipeline."""
    parsed_query = parse_query(query, backend=query_backend)
    text_features = alignment_encoder.encode_text(parsed_query.sub_queries)
    raw_score_matrix = compute_score_matrix(
        alignment_features=video_features["alignment_features"],
        text_features=text_features,
        smooth_kernel=1,
    )

    if use_qc_fr:
        feature_sets = query_conditioned_feature_refinement(
            visual_features=video_features["visual_features"],
            score_matrix=raw_score_matrix,
            context_distance=context_distance,
            qc_lambda=qc_lambda,
        )
        method = "full"
    else:
        feature_sets = l2_normalize(video_features["visual_features"])
        method = "bu_pg"

    start, end, proposal_info = bottom_up_proposal_generation(
        feature_sets=feature_sets,
        raw_score_matrix=raw_score_matrix,
        times=video_features["times"],
        duration=video_features["duration"],
        proposal_k=proposal_k,
        proposal_method=proposal_method,
        require_overlap=require_overlap,
        min_proposal_snippets=min_proposal_snippets,
    )
    scores = fuse_subquery_scores(
        score_matrix=raw_score_matrix,
        parsed_query=parsed_query,
        times=video_features["times"],
        duration=video_features["duration"],
    )
    return {
        "pred_start": start,
        "pred_end": end,
        "scores": scores,
        "raw_scores": raw_score_matrix,
        "subquery_scores": raw_score_matrix,
        "method": method,
        "sub_queries": parsed_query.sub_queries,
        "relations": parsed_query.as_dict()["relations"],
        "query_backend": parsed_query.backend,
        "candidate_count": proposal_info["candidate_count"],
        "candidate_score": proposal_info["candidate_score"],
        "selected_candidates": proposal_info["selected_candidates"],
        "final_candidate": proposal_info.get("final_candidate", {}),
        "combination_count": proposal_info.get("combination_count", 0),
        "overlap_candidate_count": proposal_info.get("overlap_candidate_count", 0),
        "proposal_method": proposal_method,
        "proposal_k": proposal_k,
        "min_proposal_snippets": proposal_info.get("min_proposal_snippets", ""),
        "overlap_required": require_overlap,
        "qc_lambda": qc_lambda if use_qc_fr else "",
        "context_distance": context_distance if use_qc_fr else "",
    }


def retrieve_moment_from_features(
    video_features,
    query,
    alignment_encoder,
    smooth_kernel=5,
    threshold_ratio=0.75,
    method="baseline",
    query_backend="spacy",
    qc_lambda=DEFAULT_QC_LAMBDA,
    context_distance=DEFAULT_CONTEXT_DISTANCE,
    proposal_k=DEFAULT_PROPOSAL_K,
    proposal_method=DEFAULT_PROPOSAL_METHOD,
    require_overlap=True,
    min_proposal_snippets=DEFAULT_MIN_PROPOSAL_SNIPPETS,
):
    if method == "baseline":
        return retrieve_baseline_from_features(
            video_features=video_features,
            query=query,
            alignment_encoder=alignment_encoder,
            smooth_kernel=smooth_kernel,
            threshold_ratio=threshold_ratio,
        )
    if method == "query_decomp":
        return retrieve_query_decomp_from_features(
            video_features=video_features,
            query=query,
            alignment_encoder=alignment_encoder,
            smooth_kernel=smooth_kernel,
            threshold_ratio=threshold_ratio,
            query_backend=query_backend,
        )
    if method == "qc_fr":
        return retrieve_qc_fr_from_features(
            video_features=video_features,
            query=query,
            alignment_encoder=alignment_encoder,
            qc_lambda=qc_lambda,
            context_distance=context_distance,
            proposal_k=proposal_k,
            proposal_method=proposal_method,
            min_proposal_snippets=min_proposal_snippets,
        )
    if method == "bu_pg":
        return retrieve_bu_pg_from_features(
            video_features=video_features,
            query=query,
            alignment_encoder=alignment_encoder,
            query_backend=query_backend,
            use_qc_fr=False,
            proposal_k=proposal_k,
            proposal_method=proposal_method,
            require_overlap=require_overlap,
            min_proposal_snippets=min_proposal_snippets,
        )
    if method == "full":
        return retrieve_bu_pg_from_features(
            video_features=video_features,
            query=query,
            alignment_encoder=alignment_encoder,
            query_backend=query_backend,
            use_qc_fr=True,
            qc_lambda=qc_lambda,
            context_distance=context_distance,
            proposal_k=proposal_k,
            proposal_method=proposal_method,
            require_overlap=require_overlap,
            min_proposal_snippets=min_proposal_snippets,
        )
    raise ValueError(f"Unsupported retrieval method: {method}")
