"""Run multi-method comparison experiments with shared model/video caches."""

from pathlib import Path
import argparse
import csv
import json

from eval import parse_limit, run_eval
from model import DEFAULT_ALIGNMENT_MODEL, SigLIPEncoder, build_visual_feature_extractor, default_i3d_checkpoint
from retrieval import PROPOSAL_METHODS, SUPPORTED_METHODS


DEFAULT_METHODS = SUPPORTED_METHODS


def write_metrics_csv(path, rows):
    """Write one summary row per method."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_json(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def run_comparison(
    project_root,
    methods=None,
    limit=100,
    num_frames=32,
    snippet_frames=16,
    smooth_kernel=5,
    threshold_ratio=0.75,
    alignment_model_name=DEFAULT_ALIGNMENT_MODEL,
    i3d_checkpoint=None,
    i3d_num_classes=400,
    i3d_batch_size=4,
    query_backend="rule",
    qc_lambda=0.5,
    context_distance=2,
    proposal_k=6,
    proposal_method="kmeans",
    require_overlap=True,
    min_proposal_snippets=0,
    experiment_name=None,
    use_feature_cache=True,
):
    """Run all ablations with shared model instances and shared video feature cache."""
    project_root = Path(project_root)
    if i3d_checkpoint is None:
        i3d_checkpoint = default_i3d_checkpoint(project_root)

    methods = methods or DEFAULT_METHODS
    limit_label = "full" if limit is None else str(limit)
    experiment_name = experiment_name or (
        f"comparison_l{limit_label}_s{num_frames}_c{snippet_frames}_i3d"
        f"_align_{Path(alignment_model_name).name}_qcl{qc_lambda}_ln{context_distance}"
        f"_k{proposal_k}_{proposal_method}_minp{min_proposal_snippets}"
    )

    prediction_dir = project_root / "outputs" / "predictions" / experiment_name
    metrics_dir = project_root / "outputs" / "experiments"
    feature_cache_dir = project_root / "outputs" / "cache" / "video_features" if use_feature_cache else None

    alignment_encoder = SigLIPEncoder(model_name=alignment_model_name)
    visual_extractor = build_visual_feature_extractor(
        i3d_checkpoint=i3d_checkpoint,
        i3d_num_classes=i3d_num_classes,
        i3d_batch_size=i3d_batch_size,
    )
    # Reuse the same heavy model objects and encoded video features for every
    # method so an ablation run is practical on a laptop GPU.
    video_cache = {}
    metrics_rows = []

    for method in methods:
        print(f"\n=== Running {method} ===")
        output_path = prediction_dir / f"{method}.csv"
        metrics, _ = run_eval(
            project_root=project_root,
            output_path=output_path,
            limit=limit,
            num_frames=num_frames,
            snippet_frames=snippet_frames,
            smooth_kernel=smooth_kernel,
            threshold_ratio=threshold_ratio,
            alignment_model_name=alignment_model_name,
            i3d_checkpoint=i3d_checkpoint,
            i3d_num_classes=i3d_num_classes,
            i3d_batch_size=i3d_batch_size,
            method=method,
            query_backend=query_backend,
            qc_lambda=qc_lambda,
            context_distance=context_distance,
            proposal_k=proposal_k,
            proposal_method=proposal_method,
            require_overlap=require_overlap,
            min_proposal_snippets=min_proposal_snippets,
            alignment_encoder=alignment_encoder,
            visual_extractor=visual_extractor,
            video_cache=video_cache,
            feature_cache_dir=feature_cache_dir,
        )
        metrics_rows.append(
            {
                "method": method,
                "limit": limit_label,
                "num_frames": num_frames,
                "snippet_frames": snippet_frames,
                "smooth_kernel": smooth_kernel,
                "threshold_ratio": threshold_ratio,
                "visual_backbone": "i3d",
                "i3d_checkpoint": str(i3d_checkpoint) if i3d_checkpoint else "",
                "i3d_num_classes": i3d_num_classes,
                "i3d_batch_size": i3d_batch_size,
                "alignment_model_name": alignment_model_name,
                "query_backend": query_backend,
                "qc_lambda": qc_lambda,
                "context_distance": context_distance,
                "proposal_k": proposal_k,
                "proposal_method": proposal_method,
                "min_proposal_snippets": min_proposal_snippets,
                "require_overlap": require_overlap,
                **metrics,
            }
        )

    metrics_csv = metrics_dir / f"{experiment_name}_metrics.csv"
    metrics_json = metrics_dir / f"{experiment_name}_metrics.json"
    write_metrics_csv(metrics_csv, metrics_rows)
    write_metrics_json(metrics_json, metrics_rows)

    print(f"\nSaved metrics: {metrics_csv}")
    print(f"Saved metrics JSON: {metrics_json}")
    print(f"Saved predictions: {prediction_dir}")
    return metrics_rows


def parse_args():
    """Command-line interface used by README and the comparison notebook."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--limit", default="100")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--snippet-frames", type=int, default=16)
    parser.add_argument("--smooth-kernel", type=int, default=5)
    parser.add_argument("--threshold-ratio", type=float, default=0.75)
    parser.add_argument("--alignment-model-name", default=DEFAULT_ALIGNMENT_MODEL)
    parser.add_argument("--i3d-checkpoint", default=str(default_i3d_checkpoint(root)))
    parser.add_argument("--i3d-num-classes", type=int, default=400)
    parser.add_argument("--i3d-batch-size", type=int, default=4)
    parser.add_argument("--query-backend", choices=["rule", "auto"], default="rule")
    parser.add_argument("--qc-lambda", type=float, default=0.5)
    parser.add_argument("--context-distance", type=int, default=2)
    parser.add_argument("--proposal-k", type=int, default=6)
    parser.add_argument(
        "--min-proposal-snippets",
        type=int,
        default=0,
        help="Minimum proposal width in snippets; 0 uses the Charades-STA auto value.",
    )
    parser.add_argument(
        "--proposal-method",
        choices=PROPOSAL_METHODS,
        default="kmeans",
    )
    parser.add_argument("--allow-disjoint-combinations", action="store_true")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--no-feature-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_comparison(
        project_root=args.project_root,
        methods=args.methods,
        limit=parse_limit(args.limit),
        num_frames=args.num_frames,
        snippet_frames=args.snippet_frames,
        smooth_kernel=args.smooth_kernel,
        threshold_ratio=args.threshold_ratio,
        alignment_model_name=args.alignment_model_name,
        i3d_checkpoint=args.i3d_checkpoint,
        i3d_num_classes=args.i3d_num_classes,
        i3d_batch_size=args.i3d_batch_size,
        query_backend=args.query_backend,
        qc_lambda=args.qc_lambda,
        context_distance=args.context_distance,
        proposal_k=args.proposal_k,
        proposal_method=args.proposal_method,
        require_overlap=not args.allow_disjoint_combinations,
        min_proposal_snippets=args.min_proposal_snippets,
        experiment_name=args.experiment_name,
        use_feature_cache=not args.no_feature_cache,
    )
