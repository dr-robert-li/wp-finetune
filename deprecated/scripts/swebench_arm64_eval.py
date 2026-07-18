#!/usr/bin/env python3
"""arm64 SWE-bench eval wrapper — bypasses the swebench CLI's hardcoded x86_64.

`python -m swebench.harness.run_evaluation` never exposes an --arch flag: its
`main()` calls `make_test_spec(instance, namespace=..., ...)` with no `arch`
kwarg, so `arch` silently defaults to `"x86_64"` inside every TestSpec it
builds (swebench.harness.test_spec.test_spec.make_test_spec, confirmed by
reading the installed package source, swebench==4.1.0).

This wrapper builds TestSpec objects directly with `arch="arm64"`, then hands
those already-built TestSpecs to `build_env_images` / `run_instances` instead
of raw dataset dicts. Both entry points are safe to call this way because:
  - `get_test_specs_from_dataset` (used by build_env_images) early-returns the
    input unchanged when `isinstance(dataset[0], TestSpec)`.
  - `run_instances` calls `make_test_spec(instance, ...)` per item, and
    `make_test_spec` early-returns `instance` unchanged when it is already a
    TestSpec — so the `arch="arm64"` baked into each spec here is never
    overwritten back to the x86_64 default.

namespace=None forces a LOCAL build (php:{version} official multi-arch image
pulled --platform=linux/arm64/v8, then repo/env/instance layers built on top)
rather than a namespace pull of a prebuilt (x86_64-only) remote image.

Patch application happens entirely inside the harness's per-instance Docker
container (swebench.harness.run_evaluation.run_instance) — this wrapper never
git-applies a model patch against the host filesystem.

Usage:
    python3 scripts/swebench_arm64_eval.py \
        --dataset SWE-bench/SWE-bench_Multilingual --split test \
        --instance_ids briannesbitt__carbon-3103 briannesbitt__carbon-3098 \
        --predictions_path gold --run_id arm64_probe1

    # Real predictions (plan 17-03 reuse):
    python3 scripts/swebench_arm64_eval.py \
        --dataset SWE-bench/SWE-bench_Lite \
        --predictions_path output/bench17/swebench_predictions.jsonl \
        --run_id lite300_v1
"""
import argparse
import json

import docker

from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import build_env_images
from swebench.harness.run_evaluation import run_instances
from swebench.harness.reporting import make_run_report
from swebench.harness.utils import load_swebench_dataset, get_predictions_from_file


def build_arm64_specs(instances):
    specs = [make_test_spec(i, arch="arm64", namespace=None) for i in instances]
    for s in specs:
        assert s.arch == "arm64" and s.platform == "linux/arm64/v8", s
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="SWE-bench/SWE-bench_Multilingual")
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--predictions_path",
        required=True,
        help="'gold' to use gold patches, or a path to a .json/.jsonl predictions file",
    )
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--instance_ids", nargs="+", default=None)
    ap.add_argument("--max_workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--force_rebuild", action="store_true")
    args = ap.parse_args()

    predictions = get_predictions_from_file(
        args.predictions_path, args.dataset, args.split
    )
    predictions = {p[KEY_INSTANCE_ID]: p for p in predictions}

    full_dataset = load_swebench_dataset(args.dataset, args.split, args.instance_ids)
    # Mirror upstream run_evaluation.get_dataset_from_preds: exclude empty-patch
    # predictions from the container runs (make_run_report still counts them as
    # empty_patch_ids / not-resolved from the full predictions dict) and skip
    # instances whose per-instance report already exists (resume-safe).
    empty_patch_ids = {
        k for k, v in predictions.items() if not v.get("model_patch")
    }
    from swebench.harness.constants import RUN_EVALUATION_LOG_DIR, LOG_REPORT

    completed_ids = {
        iid
        for iid, p in predictions.items()
        if (
            RUN_EVALUATION_LOG_DIR
            / args.run_id
            / p["model_name_or_path"].replace("/", "__")
            / iid
            / LOG_REPORT
        ).exists()
    }
    if empty_patch_ids:
        print(f"Skipping {len(empty_patch_ids)} empty-patch instance(s) (scored unresolved in report).")
    if completed_ids:
        print(f"Skipping {len(completed_ids)} already-completed instance(s) (resume).")
    instances = [
        i
        for i in full_dataset
        if i[KEY_INSTANCE_ID] in predictions
        and i[KEY_INSTANCE_ID] not in empty_patch_ids
        and i[KEY_INSTANCE_ID] not in completed_ids
    ]
    if not any(i[KEY_INSTANCE_ID] in predictions for i in full_dataset):
        raise SystemExit("No matching instances between dataset and predictions.")

    if instances:
        print(f"Building arm64 test specs for {len(instances)} instance(s)...")
        test_specs = build_arm64_specs(instances)

        client = docker.from_env()
        print("Building env images (native arm64, local build, namespace=None)...")
        build_env_images(client, test_specs, args.force_rebuild, args.max_workers)

        print(f"Running {len(instances)} instance(s)...")
        run_instances(
            predictions,
            test_specs,
            cache_level="env",
            clean=False,
            force_rebuild=args.force_rebuild,
            max_workers=args.max_workers,
            run_id=args.run_id,
            timeout=args.timeout,
            namespace=None,
        )
    else:
        print("Nothing left to run; producing report only.")

    # client=None here: make_run_report's client-based bookkeeping recomputes
    # test specs with the x86_64 default (it doesn't accept an arch override),
    # which would report our arm64 image names as "not found" — cosmetic only,
    # skipped to avoid a misleading unremoved_images/unstopped_containers count.
    report_path = make_run_report(predictions, full_dataset, args.run_id, client=None)
    print(f"Report: {report_path}")
    print(json.dumps(json.loads(report_path.read_text()), indent=2))


if __name__ == "__main__":
    main()
