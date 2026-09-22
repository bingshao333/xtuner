# Copyright (c) OpenMMLab. All rights reserved.
"""Token length statistics for XTuner v1 text datasets."""

import argparse
import csv
import logging
import os
import runpy
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


Row = dict[str, str | int | float | None]
_LENGTH_KEYS = ("num_tokens", "original_num_tokens")


def summarize_samples(name: str, shards: Sequence[Mapping[str, np.ndarray]], scope: str) -> Row:
    """Summarize original lengths and tokenization losses for one subset.

    Args:
        name (str): Dataset/subset name.
        shards (Sequence[Mapping[str, np.ndarray]]): Metadata in sample order, before packing.
        scope (str): Counting population, e.g. source samples or sampled instances.

    Returns:
        Row: One CSV row. Ratios are fractions in [0, 1], undefined metrics are None.
    """
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in _LENGTH_KEYS}
    for shard in shards:
        if "chunks" in shard:
            raise ValueError("Long-text chunk caches are not original-sample metadata")
        n = len(shard["num_tokens"])
        for key in _LENGTH_KEYS:
            values = np.asarray(shard.get(key, np.full(n, -1, dtype=np.int64)))
            # Empty ranks can promote cached counts to float. Reject non-integral values.
            if (
                values.ndim != 1
                or len(values) != n
                or values.dtype.kind not in "iuf"
                or not np.all(np.isfinite(values))
                or np.any(values != np.floor(values))
            ):
                raise ValueError(f"Invalid {key} array in subset {name!r}")
            arrays[key].append(values.astype(np.int64, copy=False))
    merged = {key: np.concatenate(vals) if vals else np.empty(0, dtype=np.int64) for key, vals in arrays.items()}
    kept, original = (merged[key] for key in _LENGTH_KEYS)
    if np.any(kept < 0) or np.any(original < -1):
        raise ValueError(f"Invalid negative token counts in subset {name!r}")
    known = original >= 0
    if np.any(original[known] < kept[known]):
        raise ValueError(f"Inconsistent tokenization truncation metadata in subset {name!r}")
    # num_tokens is recorded after tokenization truncation, before packing and collation.
    lost = original[known] - kept[known]
    known_count = int(known.sum())
    truncated_count = int((lost > 0).sum())
    lost_total = int(lost.sum())
    denominator = int(original[known].sum())
    return {
        "name": name,
        "scope": scope,
        "truncation_stage": "tokenization",
        "num_shards": len(shards),
        "count": len(kept),
        "damaged_or_empty_count": int((kept == 0).sum()),
        "known_original_count": known_count,
        "unknown_original_count": len(kept) - known_count,
        "complete_count": known_count,
        "incomplete_count": len(kept) - known_count,
        **_distribution(original[known], "original"),
        "retained_tokens_all_samples": int(kept.sum()),
        "complete_original_tokens": denominator,
        "truncated_count": truncated_count if known_count else None,
        "truncated_sample_ratio": truncated_count / known_count if known_count else None,
        "truncated_tokens": lost_total if known_count else None,
        "truncated_token_ratio": lost_total / denominator if denominator else None,
    }


def summarize_datasets(datasets: Sequence[Any]) -> list[Row]:
    """Summarize datasets after filtering and sampling.

    Args:
        datasets (Sequence[Any]): Existing datasets, each supplied once (not once per rank).

    Returns:
        list[Row]: One row per dataset name; repetitions from sample_ratio count as instances.
    """
    grouped: dict[str, list[Mapping[str, np.ndarray]]] = defaultdict(list)
    seen: set[str] = set()
    for dataset in datasets:
        path = str(Path(dataset.path).resolve())
        if path in seen:
            raise ValueError(f"Dataset supplied more than once: {path}; use sample_ratio for repetitions")
        seen.add(path)
        grouped[dataset.name].append(dataset._meta)
    return [
        summarize_samples(name, shards, "sampled_instances_before_packing") for name, shards in sorted(grouped.items())
    ]


def summarize_cache_manifest(manifest: Sequence[Mapping[str, Any]], workers: int = 1) -> list[Row]:
    """Summarize existing caches by subset.

    Args:
        manifest (Sequence[Mapping[str, Any]]): Entries with name and meta_dirs (jsonl_meta directories).
        workers (int): Parallel subset readers; only the parent writes CSVs.

    Returns:
        list[Row]: Original-source statistics before filtering and sample_ratio.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[Path] = set()
    for entry in manifest:
        for directory in entry["meta_dirs"]:
            path = Path(directory).resolve()
            if path in seen:
                raise ValueError(f"Cache supplied more than once: {path}")
            seen.add(path)
            grouped[entry["name"]].append(str(path))
    jobs = sorted(grouped.items())
    if workers == 1:
        return [_summarize_cache_subset(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_summarize_cache_subset, jobs))


def summarize_packing(packed_dataset: Any, dataloader_config: Any) -> Row:
    """Measure effective pack lengths and packing/collator losses.

    Args:
        packed_dataset (Any): Existing soft/hard packed dataset (or an unpacked ConcatDataset).
        dataloader_config (Any): Existing DataloaderConfig, supplying the real length/padding settings.

    Returns:
        Row: Global pack statistics, with padding, label shifting and dropped sequences separated.
    """
    from xtuner.v1.datasets.collator import build_text_ctx_labels
    from xtuner.v1.datasets.packing import HardPackDataset

    if dataloader_config.collator != "sft_llm_collator":
        raise ValueError("Packing statistics currently require sft_llm_collator")
    if dataloader_config.pack_level not in ("none", "soft", "hard", "__legacy"):
        raise ValueError("Preset/multimodal packing needs its own effective-token provenance and is unsupported")
    if dataloader_config.pad_token_id is None:
        raise ValueError("Set the actual padding token ID in dataloader_config.pad_token_id")
    lengths = []
    hard_pack = isinstance(packed_dataset, HardPackDataset)
    used = [np.zeros(len(dataset), dtype=np.int64) for dataset in packed_dataset.datasets] if hard_pack else []
    sequence_count = 0
    before_total = padding = partial_count = partial_tokens = dropped_count = dropped_tokens = 0
    for index in range(len(packed_dataset)):
        items = packed_dataset[index]
        if isinstance(items, dict):
            items = [items]
        before = [item["num_tokens"] for item in items]
        sequence_count += len(before)
        if hard_pack:
            infos = packed_dataset.pack_infos
            start = int(infos["indices_cu_len"][index - 1]) if index else 0
            end = int(infos["indices_cu_len"][index])
            np.add.at(used[int(infos["dataset_id"][index])], infos["indices"][start:end], before)
        ctx, _, retained = build_text_ctx_labels(
            items,
            pack_max_length=dataloader_config.pack_max_length,
            padding_token_idx=dataloader_config.pad_token_id,
            pack_to_max_length=dataloader_config.pack_to_max_length,
            pad_chunk_size=256,
        )
        after = [item["num_tokens"] for item in retained]
        losses = [old - new for old, new in zip(before, after)]
        partial_count += sum(loss > 0 for loss in losses)
        partial_tokens += sum(losses)
        dropped_count += len(before) - len(after)
        dropped_tokens += sum(before[len(after) :])
        before_total += sum(before)
        padding += ctx.num_padding
        lengths.append(ctx.input_ids.numel() - ctx.num_padding)
    # Only the final hard-pack tail loses tokens; interior slices continue in the next pack.
    input_tokens = sum(int(dataset.num_tokens.sum()) for dataset in _source_datasets(packed_dataset))
    packing_partial_count = packing_partial_tokens = packing_drop_count = packing_drop_tokens = 0
    if hard_pack:
        for dataset, retained_tokens in zip(packed_dataset.datasets, used):
            source = np.concatenate([child.num_tokens for child in _source_datasets(dataset)])
            if np.any(retained_tokens > source):
                raise ValueError("Hard-pack usage exceeds the cached sample lengths")
            partial = (retained_tokens > 0) & (retained_tokens < source)
            dropped = (retained_tokens == 0) & (source > 0)
            packing_partial_count += int(partial.sum())
            packing_partial_tokens += int((source[partial] - retained_tokens[partial]).sum())
            packing_drop_count += int(dropped.sum())
            packing_drop_tokens += int(source[dropped].sum())
    if input_tokens - before_total != packing_partial_tokens + packing_drop_tokens:
        raise ValueError("Packing token accounting does not match cached lengths; check the config/cache pairing")
    return {
        "scope": "all_packs_once_before_distributed_sampler",
        "pack_level": dataloader_config.pack_level,
        "count": len(lengths),
        **_distribution(np.asarray(lengths, dtype=np.int64), "effective"),
        "sampled_input_tokens": input_tokens,
        "tokens_before_collation": before_total,
        "packing_unassigned_tokens": input_tokens - before_total,
        "packing_truncated_instances": packing_partial_count,
        "packing_truncated_tokens": packing_partial_tokens,
        "packing_dropped_instances": packing_drop_count,
        "packing_dropped_tokens": packing_drop_tokens,
        "collator_input_sequences": sequence_count,
        "collator_truncated_sequences": partial_count,
        "collator_truncated_tokens": partial_tokens,
        "collator_truncated_sequence_ratio": partial_count / sequence_count if sequence_count else None,
        "collator_truncated_token_ratio": partial_tokens / before_total if before_total else None,
        "collator_dropped_sequences": dropped_count,
        "collator_dropped_tokens": dropped_tokens,
        "label_shift_tokens": before_total - partial_tokens - dropped_tokens - sum(lengths),
        "padding_tokens": padding,
    }


def save_reports(rows: Sequence[Row], output_dir: str | Path, packing: Row | None = None) -> tuple[Path, str]:
    """Save CSV reports in a new directory and return the log summary.

    Args:
        rows (Sequence[Row]): Final subset aggregates (also used for the log summary).
        output_dir (str | Path): Existing work/output directory.
        packing (Row | None): Optional separate packing summary.

    Returns:
        tuple[Path, str]: Output directory and short log message derived from these same rows.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_")
    run_dir = Path(tempfile.mkdtemp(prefix=f"token_stats_{stamp}", dir=output))
    _write_csv(run_dir / "subset_token_stats.csv", rows)
    if packing is not None:
        _write_csv(run_dir / "packing_token_stats.csv", [packing])
    count = sum(int(row["count"]) for row in rows)
    known = sum(int(row["known_original_count"]) for row in rows)
    complete = sum(int(row["complete_count"]) for row in rows)
    lost = sum(int(row["truncated_tokens"] or 0) for row in rows)
    truncated = sum(int(row["truncated_count"] or 0) for row in rows)
    truncation_summary = (
        f"{truncated} samples, {lost} tokens (complete coverage only)" if complete else "unknown (no complete samples)"
    )
    message = (
        f"Token statistics: {len(rows)} subsets, {count} samples/instances; original length known {known}/{count}, "
        f"complete {complete}/{count}; tokenization truncated {truncation_summary}. "
        f"CSV: {run_dir / 'subset_token_stats.csv'}"
    )
    if packing is not None:
        message += (
            f" Packing: {packing['count']} packs, {packing['effective_total_tokens']} effective tokens; "
            f"packing truncated {packing['packing_truncated_tokens']} tokens, "
            f"dropped {packing['packing_dropped_instances']} instances; "
            f"collator truncated {packing['collator_truncated_tokens']} tokens, "
            f"dropped {packing['collator_dropped_sequences']} sequences. "
            f"CSV: {run_dir / 'packing_token_stats.csv'}"
        )
    return run_dir, message


def main() -> None:
    """Run token statistics from a config or cache manifest."""
    import json

    os.environ.setdefault("TQDM_DISABLE", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache-manifest", type=Path, help="JSON list of {name, meta_dirs}; never rebuilds caches")
    source.add_argument("--config", type=Path, help="Python file defining build_stats_inputs(work_dir)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Existing work/output directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel subset readers in cache mode")
    args = parser.parse_args()
    # Avoid reporting replicated datasets once per rank.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Run this offline coordinator once, without torchrun; use --workers for parallel cache reads")
    packing = None
    if args.cache_manifest:
        manifest = json.loads(args.cache_manifest.read_text())
        for entry in manifest:
            entry["meta_dirs"] = [str(args.cache_manifest.parent / path) for path in entry["meta_dirs"]]
        rows = summarize_cache_manifest(manifest, args.workers)
        log = logging.getLogger("xtuner.token_stats")
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        from xtuner.v1.utils import get_logger, log_format

        log = get_logger()
        # Keep per-sample logs out of the report.
        log.remove()
        log.add(sys.stderr, format=log_format(), filter=lambda record: record["name"] == __name__)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        factory = runpy.run_path(str(args.config))["build_stats_inputs"]
        inputs = factory(args.output_dir)
        rows = summarize_datasets(inputs["datasets"])
        if "packed_dataset" in inputs:
            packing = summarize_packing(inputs["packed_dataset"], inputs["dataloader_config"])
    _, message = save_reports(rows, args.output_dir, packing)
    log.info(message)


def _distribution(values: np.ndarray, prefix: str) -> Row:
    return {
        f"{prefix}_total_tokens": int(values.sum()) if len(values) else None,
        f"{prefix}_mean_tokens": float(values.mean()) if len(values) else None,
        f"{prefix}_median_tokens": float(np.median(values)) if len(values) else None,
        f"{prefix}_p99_tokens": float(np.percentile(values, 99)) if len(values) else None,
        f"{prefix}_max_tokens": int(values.max()) if len(values) else None,
    }


def _summarize_cache_subset(job: tuple[str, list[str]]) -> Row:
    name, directories = job
    shards = []
    for directory in directories:
        path = Path(directory)
        if (path / "chunks.npy").exists():
            raise ValueError(f"Chunked caches have no original-document coverage: {path}")
        shard = {"num_tokens": np.load(path / "num_tokens.npy", allow_pickle=False)}
        for key in _LENGTH_KEYS[1:]:
            if (path / f"{key}.npy").exists():
                shard[key] = np.load(path / f"{key}.npy", allow_pickle=False)
        shards.append(shard)
    return summarize_samples(name, shards, "source_samples_before_filtering")


def _source_datasets(dataset: Any) -> list[Any]:
    if hasattr(dataset, "datasets"):
        return [source for child in dataset.datasets for source in _source_datasets(child)]
    return [dataset]


def _write_csv(path: Path, rows: Sequence[Row]) -> None:
    if not rows:
        raise ValueError("No subsets to report")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
