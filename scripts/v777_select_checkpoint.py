#!/usr/bin/env python3
"""Validation-only physical selection among completed DirectNet epoch snapshots."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'external_compact_models'):
    sys.path.insert(0, str(path))
from neural_network.config import CODE_TO_TECH_VARIANT, LOCAL_VARIANT_CODES
from neural_network.data.dataset import validate_canonical_dataset
from neural_network.data.normalize import normalizer_from_stats
from neural_network.data.sampling import stratified_sample_indices
from pycircuitsim.models.mosfet_directnet_full import _load_artifacts

QUANTUM = 2.0 ** -42
TAIL_FLOORS = np.asarray([200 * QUANTUM] * 4 + [1e-20] * 4)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def terminals(values: np.ndarray) -> np.ndarray:
    """Independent surfaces plus source closure, in current/charge groups."""
    return np.column_stack([values[:, :3], -values[:, :3].sum(axis=1),
                            values[:, 3:], -values[:, 3:].sum(axis=1)])


def physical_score(
    prediction: np.ndarray, truth: np.ndarray, x: np.ndarray, temperature: np.ndarray,
    polarity: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    error = np.abs(terminals(prediction) - terminals(truth))
    tails, medians, p95s = [], [], []
    for temp in np.unique(temperature):
        group = temperature == temp
        tails.append(np.quantile(error[group], [.95, .99], axis=0))
        off = group & (abs(x[:, 1] - x[:, 2]) < .02) & (polarity * (x[:, 0] - x[:, 2]) > .3)
        off &= abs(truth[:, 0]) >= 20 * QUANTUM
        if not off.any():
            raise ValueError(f'validation subset lacks resolved off-current rows at {temp}')
        if np.any(prediction[off, 0] == 0):
            medians.append(float('inf'))
            p95s.append(float('inf'))
            continue
        logs = abs(np.log10(abs(prediction[off, 0]) / abs(truth[off, 0])))
        medians.append(float(np.median(logs)))
        p95s.append(float(np.quantile(logs, .95)))
    return np.stack(tails), (max(medians), max(p95s))


def sign_errors(prediction: np.ndarray, truth: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Count resolved-current sign mistakes separately for each temperature."""
    reference = terminals(truth)[:, :4]
    predicted = terminals(prediction)[:, :4]
    wrong = (np.sign(predicted) != np.sign(reference)) & (abs(reference) >= 20 * QUANTUM)
    return np.stack([wrong[temperature == temp].sum(axis=0)
                     for temp in np.unique(temperature)])


def select(checkpoint: Path, data_path: Path, origin_path: Path, output: Path,
           tech: str, device: str, cuda: bool) -> None:
    marker_path = Path(str(checkpoint) + '.complete')
    marker = json.loads(marker_path.read_text())
    if marker['family'] != 'directnet-full':
        raise ValueError('this registered selector is DirectNet-only')
    if (marker['training']['tech_scope'] != tech
            or not checkpoint.name.endswith(f'_{device}_best.pt')):
        raise ValueError('selection technology/polarity do not match the checkpoint')
    validate_canonical_dataset(data_path)
    validate_canonical_dataset(origin_path)
    if marker['dataset_sha256'] != sha(data_path):
        raise ValueError('selection data do not match the trained bundle')
    snapshot_info = marker['training']['epoch_snapshots']
    snapshot_dir = Path(snapshot_info['directory'])
    manifest_path = snapshot_dir / snapshot_info['manifest']
    if sha(manifest_path) != snapshot_info['manifest_sha256']:
        raise ValueError('snapshot manifest checksum mismatch')
    records = json.loads(manifest_path.read_text())
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    labels = np.load(data_path.with_name(data_path.stem + '_tech_variant_labels.npy'))
    with np.load(data_path, allow_pickle=False) as data:
        parent_sha = str(data['meta_parent_dataset_sha256'])
        split_sha = str(data['meta_parent_split_sha256'])
        validation = np.flatnonzero(data['split_partition'] == 1)
        geometry = data['geometry'][validation]
        subset = stratified_sample_indices(
            np.column_stack([labels[validation], geometry[:, :3]]),
            min(50000, len(validation)), 20261003)
        indices = validation[subset]
        x, geo, truth = data['inputs'][indices], data['geometry'][indices], data['outputs'][indices]
        row_ids = data['row_ids'][indices]
        parent_rows = len(data['row_ids'])
    codes = np.array([LOCAL_VARIANT_CODES[tech][CODE_TO_TECH_VARIANT[int(c)]] for c in labels[indices]])
    # Paired origin points belong to validation geometry groups, never test.
    olab = np.load(origin_path.with_name(origin_path.stem + '_tech_variant_labels.npy'))
    polarity = 1.0 if device == 'nmos' else -1.0
    vdd = {'tsmc5': .65, 'tsmc12': .8}[tech]
    with np.load(origin_path, allow_pickle=False) as data:
        if (str(data['meta_parent_dataset_sha256']) != parent_sha
                or str(data['meta_parent_split_sha256']) != split_sha):
            raise ValueError('origin validation data have a different parent or split')
        ox_all = data['inputs']
        mask = ((data['split_partition'] == 1) & (data['row_ids'] >= parent_rows)
                & (abs(ox_all[:, 0]) == 1e-5) & (ox_all[:, 3] == 0.)
                & (polarity * ox_all[:, 1] >= .75 * vdd))
        oi = np.flatnonzero(mask)
        ox, og, oy, oids = ox_all[oi], data['geometry'][oi], data['outputs'][oi], data['row_ids'][oi]
    oc = np.array([LOCAL_VARIANT_CODES[tech][CODE_TO_TECH_VARIANT[int(c)]] for c in olab[oi]])
    pairs: dict[tuple[float, ...], dict[bool, int]] = {}
    for i in range(len(oi)):
        key = (float(olab[oi[i]]), *og[i, :3], ox[i, 1])
        pairs.setdefault(key, {})[bool(ox[i, 0] > 0)] = i
    if not pairs or any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError('validation origin pairs are incomplete')
    plus = np.asarray([pair[True] for pair in pairs.values()])
    minus = np.asarray([pair[False] for pair in pairs.values()])
    reference_gds = (oy[plus, 0] - oy[minus, 0]) / 2e-5
    positive = reference_gds > 1e-9
    model, stats, _, _ = _load_artifacts(checkpoint)
    model = copy.deepcopy(model)  # Do not mutate the shared runtime artifact cache.
    normalizer = normalizer_from_stats(stats)
    target_device = torch.device('cuda' if cuda else 'cpu')
    model.to(target_device)
    all_x = normalizer.normalize_inputs(np.concatenate([x, ox]), np.concatenate([geo, og])).astype(np.float32)
    all_codes = np.concatenate([codes, oc])
    tx, tc = torch.from_numpy(all_x).to(target_device), torch.from_numpy(all_codes).to(target_device)

    def predict() -> tuple[np.ndarray, np.ndarray]:
        values = []
        with torch.no_grad():
            for lo in range(0, len(tx), 8192):
                values.append(model(tx[lo:lo+8192], tech_codes=tc[lo:lo+8192]).cpu().numpy())
        physical = normalizer.denormalize_outputs(np.concatenate(values))
        return physical[:len(x)], physical[len(x):]

    base_prediction, base_origin = predict()
    baseline_tails, baseline_rank = physical_score(base_prediction, truth, x, geo[:, 2], polarity)
    baseline_signs = sign_errors(base_prediction, truth, geo[:, 2])
    baseline_gds = (base_origin[plus, 0] - base_origin[minus, 0]) / 2e-5
    protected = positive & (baseline_gds >= 0)
    selected_epoch = int(marker['training']['selected_epoch'])
    best = (*baseline_rank, float(marker['training']['best_validation_loss']))
    audit = []
    for record in records:
        path = snapshot_dir / record['file']
        if sha(path) != record['sha256']:
            raise ValueError(f'snapshot checksum mismatch: {path}')
        model.load_state_dict(torch.load(path, map_location=target_device, weights_only=True))
        predicted, origin = predict()
        if not np.isfinite(predicted).all() or not np.isfinite(origin).all():
            audit.append(dict(epoch=record['epoch'], eligible=False, reason='nonfinite_prediction'))
            continue
        tails, rank = physical_score(predicted, truth, x, geo[:, 2], polarity)
        gds = (origin[plus, 0] - origin[minus, 0]) / 2e-5
        eligible = (np.isfinite(rank).all() and tails.shape == baseline_tails.shape
                    and bool(np.all(tails <= 1.02 * baseline_tails + TAIL_FLOORS))
                    and bool(np.all(sign_errors(predicted, truth, geo[:, 2]) <= baseline_signs))
                    and not np.any(gds[protected] < 0))
        score = (*rank, float(record['validation_loss']))
        audit.append(dict(epoch=record['epoch'], eligible=bool(eligible),
            score=[float(v) if np.isfinite(v) else None for v in score],
            absolute_error_tails=tails.tolist(),
            sign_error_counts=sign_errors(predicted, truth, geo[:, 2]).tolist(),
            new_negative_origin_slopes=int(np.sum(gds[protected] < 0))))
        if eligible and score < best:
            best = score
            selected_epoch = int(record['epoch'])
    selected = next(record for record in records if record['epoch'] == selected_epoch)
    stem = checkpoint.name.removesuffix('_best.pt')
    destination = output / checkpoint.name
    shutil.copyfile(snapshot_dir / selected['file'], destination)
    normalization = checkpoint.with_name(stem + '_norm.npz')
    shutil.copyfile(normalization, output / normalization.name)
    selector_source = sha(Path(__file__))
    np.savez_compressed(output / 'validation_rows.npz', original_row_ids=row_ids, origin_row_ids=oids)
    selection = dict(rule='v777-physical-validation-v1', source_sha256=selector_source,
        validation_seed=20261003, validation_rows=len(x), origin_pairs=len(pairs),
        tail_floors=TAIL_FLOORS.tolist(), baseline_absolute_error_tails=baseline_tails.tolist(),
        baseline_sign_error_counts=baseline_signs.tolist(),
        validation_rows_sha256=sha(output / 'validation_rows.npz'),
        source_dataset_sha256=sha(data_path), origin_dataset_sha256=sha(origin_path),
        parent_marker_sha256=sha(marker_path), original_selected_epoch=marker['training']['selected_epoch'],
        selected_epoch=selected_epoch,
        baseline_rank=[float(v) if np.isfinite(v) else None for v in baseline_rank],
        selected_rank=[float(v) if np.isfinite(v) else None for v in best],
        inference_device=str(target_device), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    marker['checkpoint_sha256'] = sha(destination)
    marker['training'] = {**marker['training'], 'selected_epoch': selected_epoch, 'selection': selection}
    (output / (checkpoint.name + '.complete')).write_text(json.dumps(marker, indent=2, sort_keys=True))
    (output / 'selection.json').write_text(json.dumps(dict(selection=selection, epochs=audit), indent=2, allow_nan=False))
    print(json.dumps(selection, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--origin-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tech', choices=['tsmc5', 'tsmc12'], required=True)
    parser.add_argument('--device', choices=['nmos', 'pmos'], required=True)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    select(args.checkpoint, args.data, args.origin_data, args.output, args.tech, args.device, args.cuda)


if __name__ == '__main__':
    main()
