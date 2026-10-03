#!/usr/bin/env python3
"""Freeze parent splits and append an isolated, OSDI-labelled drain-origin arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'external_compact_models', ROOT / 'external_compact_models/bsim_cmg'):
    sys.path.insert(0, str(path))

from neural_network.config import CODE_TO_TECH_VARIANT, LOCAL_VARIANT_CODES
from neural_network.data.dataset import validate_canonical_dataset
from neural_network.data.sampling import _groups, grouped_split_indices, persisted_split_indices
from neural_network.eval.loo_labels import get_or_build_tech_variant_labels, write_sidecar_meta
from pycmg.nn_generate import (SAMPLE_CLASS_NAMES, SAMPLE_CLASS_CODES, OUTPUT_COLUMNS,
    _create_model_and_instance, _eval_single_point_with_reason, drain_origin_points)
from pycmg.nn_config import TECH_CONFIGS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluate_group(job: tuple[Any, ...]) -> dict[str, Any]:
    tech, device, code, geometry, old_inputs, partition = job
    _, variant = CODE_TO_TECH_VARIANT[int(code)]
    nfin, length, temp = map(float, geometry[:3])
    config = TECH_CONFIGS[tech]
    built = _create_model_and_instance(config, device, variant, length, nfin, temp)
    if built is None:
        raise RuntimeError(f'origin group cannot be evaluated: {tech}/{device}/{variant}/{geometry[:3]}')
    _, instance, process = built
    expected = np.array([nfin, length, temp, *process.as_array()])
    if not np.allclose(geometry, expected, rtol=1e-12, atol=1e-18):
        raise ValueError('parent process fingerprint differs from the origin evaluator')
    existing = {tuple(row) for row in old_inputs.tolist()}
    proposed = drain_origin_points(config.vdd, device == 'pmos')
    inputs = np.asarray([row for row in proposed if tuple(row) not in existing])
    outputs = []
    for vd, vg, vs, vb in inputs:
        result, reason = _eval_single_point_with_reason(
            instance, vd=float(vd), vg=float(vg), vs=float(vs), vb=float(vb), _silent=True)
        if result is None:
            raise RuntimeError(f'origin evaluation rejected {tech}/{variant}/{nfin}/{length}/{temp}: {reason}')
        outputs.append([result[name] for name in OUTPUT_COLUMNS])
    card = Path(config.resolve_modelcard(device, variant, length, nfin)).resolve()
    return dict(inputs=inputs, geometry=np.tile(geometry, (len(inputs), 1)),
        outputs=np.asarray(outputs, dtype=np.float64), code=int(code), partition=int(partition),
        card=str(card), card_sha256=sha256(card),
        manifest=dict(tech=tech, device=device, variant=variant, L=length, NFIN=nfin,
            temperature_k=temp, status='complete', requested=len(inputs), kept=len(inputs),
            rejected=0, failure_reason_counts={}, sample_class='drain_origin'))


def write_dataset(path: Path, arrays: dict[str, np.ndarray], labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix('.npz.complete').exists():
        raise FileExistsError(path)
    temporary = path.with_name(f'{path.stem}.{os.getpid()}.partial.npz')
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    np.save(path.with_name(path.stem + '_tech_variant_labels.npy'), labels)
    write_sidecar_meta(path, arrays['geometry'], labels)
    path.with_suffix('.npz.complete').write_text(json.dumps(dict(
        dataset=path.name, dataset_sha256=sha256(path), rows=len(labels),
        generator_release='V7.7.7', source_commit=str(arrays['meta_source_commit']),
        source_dirty=False), indent=2, sort_keys=True))
    validate_canonical_dataset(path)


def prepare(parent: Path, output_root: Path, tech: str, device: str, workers: int) -> None:
    validate_canonical_dataset(parent)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('origin preparation requires a clean committed worktree')
    paths = [output_root / arm / parent.name for arm in ('control', 'origin')]
    if any(path.exists() for path in paths):
        raise FileExistsError('prepared arm already exists; use an isolated output root')
    with np.load(parent, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    labels = get_or_build_tech_variant_labels(str(parent), device)
    if any(CODE_TO_TECH_VARIANT[int(code)][0] != tech for code in np.unique(labels)):
        raise ValueError('parent labels do not match the selected technology')
    local = np.array([LOCAL_VARIANT_CODES[tech][CODE_TO_TECH_VARIANT[int(code)]]
                      for code in labels], dtype=np.int64)
    strata = np.column_stack([local, arrays['geometry'][:, :3]])
    if 'split_partition' in arrays or 'split_rank' in arrays:
        raise ValueError('prepare the preserved parent once; nested preparation is not supported')
    splits = grouped_split_indices(strata, 0.8, 0.1, 42)
    n_rows = len(labels)
    partition = np.empty(n_rows, dtype=np.int8)
    rank = np.empty(n_rows, dtype=np.int64)
    for index, split in enumerate(splits):
        partition[split] = index
        rank[split] = np.arange(len(split))
    groups = _groups(strata)
    jobs = [(tech, device, int(labels[group[0]]), arrays['geometry'][group[0]],
             arrays['inputs'][group], int(partition[group[0]])) for group in groups]
    split_hash = hashlib.sha256(partition.tobytes() + rank.tobytes()).hexdigest()
    arrays.update(row_ids=np.arange(n_rows, dtype=np.int64),
        split_partition=partition, split_rank=rank,
        meta_parent_dataset=np.array(parent.name), meta_parent_dataset_sha256=np.array(sha256(parent)),
        meta_parent_source_commit=arrays['meta_source_commit'],
        meta_source_commit=np.array(commit), meta_source_dirty=np.array(False),
        meta_generator_release=np.array('V7.7.7'),
        meta_split_rule=np.array('frozen-parent-combo-v1'), meta_split_seed=np.array(42),
        meta_parent_split_sha256=np.array(split_hash),
        meta_generator_command=np.array(' '.join(sys.argv)),
        meta_sample_class_names=np.asarray(SAMPLE_CLASS_NAMES))
    write_dataset(paths[0], arrays, labels)
    print(f'{tech}/{device}: frozen {n_rows} rows, {len(groups)} groups', flush=True)
    with mp.get_context('spawn').Pool(workers) as pool:
        results = []
        for result in pool.imap(evaluate_group, jobs):
            results.append(result)
            if len(results) % 25 == 0:
                print(f'{tech}/{device}: origin {len(results)}/{len(jobs)} groups', flush=True)
    overlay_rows = sum(len(result['inputs']) for result in results)
    counts = [len(split) for split in splits]
    overlay_partition, overlay_rank, overlay_labels = [], [], []
    for result in results:
        count = len(result['inputs'])
        code = result['partition']
        overlay_partition.append(np.full(count, code, dtype=np.int8))
        overlay_rank.append(np.arange(counts[code], counts[code] + count, dtype=np.int64))
        overlay_labels.append(np.full(count, result['code'], dtype=labels.dtype))
        counts[code] += count
    for name in ('inputs', 'geometry', 'outputs'):
        arrays[name] = np.concatenate([arrays[name], *[r[name] for r in results]])
    arrays['sample_class'] = np.concatenate([arrays['sample_class'],
        np.full(overlay_rows, SAMPLE_CLASS_CODES['drain_origin'], dtype=np.int8)])
    arrays['row_ids'] = np.arange(n_rows + overlay_rows, dtype=np.int64)
    arrays['split_partition'] = np.concatenate([partition, *overlay_partition])
    arrays['split_rank'] = np.concatenate([rank, *overlay_rank])
    labels = np.concatenate([labels, *overlay_labels])
    for name in ('requested_rows', 'kept_rows'):
        arrays['meta_' + name] = np.array(int(arrays['meta_' + name]) + overlay_rows)
    manifest = json.loads(str(arrays['meta_manifest_json']))
    manifest.extend(result['manifest'] for result in results)
    cards = json.loads(str(arrays['meta_modelcard_sha256_json']))
    cards.update({r['card']: r['card_sha256'] for r in results})
    arrays.update(meta_manifest_json=np.array(json.dumps(manifest, sort_keys=True)),
        meta_modelcard_sha256_json=np.array(json.dumps(cards, sort_keys=True)),
        meta_dataset_variant=np.array(str(arrays['meta_dataset_variant']) + '_plus_drain_origin'),
        meta_origin_rows=np.array(overlay_rows))
    # Every parent row retains both membership and order; every new row stays
    # in its existing geometry group. This is checked before publication.
    restored = persisted_split_indices(np.column_stack([labels, arrays['geometry'][:, :3]]),
                                       arrays['split_partition'], arrays['split_rank'])
    for old, new in zip(splits, restored):
        if not np.array_equal(old, new[new < n_rows]):
            raise AssertionError('parent split order changed')
    write_dataset(paths[1], arrays, labels)
    print(f'{tech}/{device}: {overlay_rows} origin rows; parent split/order preserved', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--tech', choices=['tsmc5', 'tsmc6', 'tsmc7', 'tsmc12', 'tsmc16'], required=True)
    parser.add_argument('--device', choices=['nmos', 'pmos'], required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    prepare(args.parent.resolve(), args.output_root.resolve(), args.tech, args.device, args.workers)


if __name__ == '__main__':
    main()
