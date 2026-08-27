"""Dataset wrapper and loader for BSIMAR / DirectNet training.

One loader for both architectures. The caller picks ``norm_mode``:
``"zscore"`` for DirectNet, ``"asinh"`` for the Transformer.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from neural_network.data.contracts import CANONICAL_SAFETY_REJECTION_REASONS
from neural_network.data.normalize import _NormalizerBase, normalizer_for
from neural_network.data.sampling import (
    grouped_split_indices,
    stratified_sample_indices,
)


class MOSFETDataset(Dataset):
    """(inputs, outputs, tech_codes) tuples in normalised space.

    ``sample_class`` (V6.4.7 S9b): optional per-row generator origin tag
    (int8 codes from PyCMG ``nn_generate.SAMPLE_CLASS_CODES``), kept
    aligned with the split rows. ``sample_class_names`` maps code → name
    (index = code). Both are metadata only — ``__getitem__`` is unchanged.
    """

    def __init__(
        self,
        inputs_norm: np.ndarray,
        outputs_norm: np.ndarray,
        tech_codes: np.ndarray,
        sample_class: Optional[np.ndarray] = None,
        sample_class_names: Optional[List[str]] = None,
    ) -> None:
        self.inputs = torch.tensor(inputs_norm, dtype=torch.float32)
        self.outputs = torch.tensor(outputs_norm, dtype=torch.float32)
        self.tech_codes = torch.tensor(tech_codes, dtype=torch.long)
        self.sample_class = (
            torch.tensor(sample_class, dtype=torch.int8)
            if sample_class is not None else None)
        self.sample_class_names = sample_class_names

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.outputs[idx], self.tech_codes[idx]


# Drop rows below the modelcard noise floor for Id. Charges and caps are
# absorbed by the asinh per-target scale, so the only useful filter is on
# Id (v5 plan §4-B4).
DEFAULT_FILTER_THRESHOLDS: Dict[str, float] = {"id": 1e-15}

# Legacy sample_class code for rows of unknown origin — mirrors
# PyCMG nn_generate._assemble, which tags pre-B1 bins as "lhs" (code 6).
_LEGACY_LHS_CLASS_CODE: int = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_canonical_dataset(data_path: Union[str, Path]) -> None:
    """Reject incomplete, diagnostic, stale, or untraceable training data."""
    path = Path(data_path)
    marker_path = path.with_suffix(path.suffix + ".complete")
    if not marker_path.is_file():
        raise ValueError(f"dataset completion marker is missing: {marker_path}")
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid dataset completion marker: {marker_path}") from exc
    if not isinstance(marker, dict):
        raise ValueError(f"invalid dataset completion marker: {marker_path}")
    if marker.get("dataset") != path.name:
        raise ValueError("dataset completion marker names a different file")
    if marker.get("dataset_sha256") != _sha256(path):
        raise ValueError("dataset checksum does not match completion marker")

    required = {
        "meta_allow_rejected_points", "meta_dataset_variant",
        "meta_dropped_bins", "meta_generator_release", "meta_kept_rows",
        "meta_manifest_json", "meta_modelcard_sha256_json",
        "meta_osdi_sha256", "meta_rejected_rows", "meta_requested_rows",
        "meta_source_commit", "meta_source_dirty",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                "dataset provenance metadata is incomplete: " + ", ".join(missing)
            )

        def _scalar(name: str) -> object:
            value = np.asarray(data[name])
            if value.size != 1:
                raise ValueError(f"dataset metadata {name} must be scalar")
            return value.reshape(()).item()

        rows = len(data["outputs"])
        requested = int(_scalar("meta_requested_rows"))
        kept = int(_scalar("meta_kept_rows"))
        rejected = int(_scalar("meta_rejected_rows"))
        dropped = int(_scalar("meta_dropped_bins"))
        allow_diagnostic = bool(_scalar("meta_allow_rejected_points"))
        allow_safety = (
            bool(_scalar("meta_allow_safety_rejections"))
            if "meta_allow_safety_rejections" in data.files else False
        )
        if allow_diagnostic or dropped or (rejected and not allow_safety):
            raise ValueError(
                "diagnostic dataset cannot be used for canonical training"
            )
        if requested != kept + rejected or kept != rows:
            raise ValueError("dataset row counts do not prove complete generation")
        if bool(_scalar("meta_source_dirty")):
            raise ValueError("dataset was generated from a dirty source tree")
        source_commit = str(_scalar("meta_source_commit"))
        release = str(_scalar("meta_generator_release"))
        osdi_sha = str(_scalar("meta_osdi_sha256"))
        try:
            modelcard_hashes = json.loads(
                str(_scalar("meta_modelcard_sha256_json"))
            )
            manifest = json.loads(str(_scalar("meta_manifest_json")))
        except json.JSONDecodeError as exc:
            raise ValueError("dataset hash or bin manifest JSON is invalid") from exc
        sha_pattern = re.compile(r"[0-9a-f]{64}")
        if not sha_pattern.fullmatch(osdi_sha):
            raise ValueError("dataset OSDI SHA-256 is invalid")
        if not isinstance(modelcard_hashes, dict) or not modelcard_hashes or any(
            not isinstance(value, str) or not sha_pattern.fullmatch(value)
            for value in modelcard_hashes.values()
        ):
            raise ValueError("dataset modelcard SHA-256 map is invalid")
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("dataset bin manifest is empty or invalid")
        if rejected:
            manifest_rejected = 0
            manifest_reasons: Set[str] = set()
            for entry in manifest:
                if not isinstance(entry, dict):
                    raise ValueError("dataset bin manifest is invalid")
                entry_rejected = int(entry.get("rejected", 0))
                manifest_rejected += entry_rejected
                reason_counts = entry.get("failure_reason_counts", {})
                if not isinstance(reason_counts, dict):
                    raise ValueError("dataset rejection manifest is invalid")
                try:
                    counted_rejections = sum(
                        int(count) for count in reason_counts.values()
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "dataset rejection manifest is invalid"
                    ) from exc
                if counted_rejections != entry_rejected:
                    raise ValueError("dataset rejection manifest is invalid")
                manifest_reasons.update(str(reason) for reason in reason_counts)
            unexpected = sorted(
                manifest_reasons.difference(CANONICAL_SAFETY_REJECTION_REASONS)
            )
            if manifest_rejected != rejected or unexpected:
                raise ValueError(
                    "dataset safety-rejection provenance is invalid"
                )
        if marker.get("rows") != rows:
            raise ValueError("dataset row count does not match completion marker")
        if marker.get("source_commit") != source_commit:
            raise ValueError("dataset source commit does not match completion marker")
        if marker.get("generator_release") != release:
            raise ValueError("dataset release does not match completion marker")
        if (not re.fullmatch(r"[0-9a-f]{40}", source_commit)
                or not re.fullmatch(r"V\d+\.\d+\.\d+", release)):
            raise ValueError("dataset provenance contains unknown values")


def filter_small_targets(
    outputs: np.ndarray,
    column_names: List[str],
    thresholds: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    thresholds = thresholds or DEFAULT_FILTER_THRESHOLDS
    mask = np.ones(len(outputs), dtype=bool)
    for i, name in enumerate(column_names):
        if name in thresholds:
            mask &= np.abs(outputs[:, i]) > thresholds[name]
    return mask


def load_and_split_bsimar(
    data_path: str,
    column_names: List[str],
    device_type: str,
    norm_mode: str = "asinh",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    apply_filter: bool = True,
    filter_thresholds: Optional[Dict[str, float]] = None,
    exclude_techs: Optional[Set[str]] = None,
    max_rows: Optional[int] = None,
    output_subset: Optional[List[str]] = None,
    tech_scope: str = "universal",
    split_mode: str = "combo",
) -> Tuple[MOSFETDataset, MOSFETDataset, MOSFETDataset, _NormalizerBase]:
    """Load .npz, label, optionally filter / exclude techs / cap, split, normalise.

    ``output_subset``: optional list of column names from
    ``OUTPUT_COLUMN_ORDER``. If given, only those columns are kept on
    every split, and the normalizer's stats are sized to the subset.

    ``tech_scope``: one of ``"universal"`` (no remap; default) or a
    per-tech scope name (``"tsmc5"`` / ``"tsmc7"``). When non-universal,
    each row's tech_code is remapped from the universal vocab to a
    0-indexed local vocab whose size matches the trained per-tech
    embedding. Rows outside the scope (which should already be removed
    by ``exclude_techs``) collapse to the local UNKNOWN slot at the tail.
    """
    from neural_network.eval.loo_labels import get_or_build_tech_variant_labels

    data = np.load(data_path, allow_pickle=True)
    inputs = data["inputs"]
    geometry = data["geometry"]
    outputs = data["outputs"]
    declared_columns = (
        [str(value) for value in data["meta_output_columns"]]
        if "meta_output_columns" in data.files else list(column_names)
    )
    if declared_columns != list(column_names):
        raise ValueError(
            "dataset declared output columns do not match the requested "
            f"training contract: declared={declared_columns}, "
            f"requested={list(column_names)}"
        )
    if outputs.ndim != 2 or outputs.shape[1] != len(declared_columns):
        raise ValueError(
            "dataset output width does not match its declared output columns"
        )
    tech_codes = get_or_build_tech_variant_labels(
        data_path, device_type, verbose=True)

    # V6.4.7 S9b: per-row generator origin tag, kept aligned through every
    # row operation below. Missing → legacy "lhs" code (nn_generate
    # convention for pre-B1 bins).
    if "sample_class" in data.files:
        sample_class = np.asarray(data["sample_class"], dtype=np.int8)
    else:
        print(f"  [warn] {data_path} has no sample_class — tagging all "
              f"rows as legacy 'lhs' (code {_LEGACY_LHS_CLASS_CODE})")
        sample_class = np.full(
            len(outputs), _LEGACY_LHS_CLASS_CODE, dtype=np.int8)
    sample_class_names: Optional[List[str]] = None
    if "meta_sample_class_names" in data.files:
        sample_class_names = [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in data["meta_sample_class_names"]
        ]

    n0 = len(outputs)
    if apply_filter:
        keep = filter_small_targets(outputs, column_names, filter_thresholds)
        inputs, geometry, outputs = inputs[keep], geometry[keep], outputs[keep]
        tech_codes = tech_codes[keep]
        sample_class = sample_class[keep]
        n_drop = n0 - len(outputs)
        print(f"  Filter Id>1e-15: {n0} -> {len(outputs)} "
              f"(dropped {n_drop} rows, {100.0 * n_drop / max(n0, 1):.2f}%)")
    else:
        print(f"  Filter OFF: keeping all {n0} rows "
              "(small-Id rows retained)")

    if exclude_techs:
        from neural_network.config import TECH_VARIANT_CODES
        excl = {
            code for (tech, _), code in TECH_VARIANT_CODES.items()
            if tech in exclude_techs
        }
        keep = np.array(
            [int(c) not in excl for c in tech_codes], dtype=bool)
        inputs, geometry, outputs = inputs[keep], geometry[keep], outputs[keep]
        tech_codes = tech_codes[keep]
        sample_class = sample_class[keep]
        print(f"  Excluded {exclude_techs}: kept {keep.sum()} samples")

    if max_rows is not None and len(outputs) > max_rows:
        cap_strata = np.column_stack(
            [tech_codes, geometry[:, :3], sample_class]
        )
        idx = stratified_sample_indices(cap_strata, max_rows, seed)
        inputs, geometry, outputs = inputs[idx], geometry[idx], outputs[idx]
        tech_codes = tech_codes[idx]
        sample_class = sample_class[idx]
        print(f"  Capped to {max_rows} rows")

    if tech_scope != "universal":
        from neural_network.config import (
            CODE_TO_TECH_VARIANT,
            LOCAL_VARIANT_CODES,
            LOCAL_UNKNOWN_CODE_ID,
            VALID_TECH_SCOPES,
        )
        if tech_scope not in LOCAL_VARIANT_CODES:
            raise ValueError(
                f"tech_scope {tech_scope!r} not in {VALID_TECH_SCOPES}")
        local_table = LOCAL_VARIANT_CODES[tech_scope]
        unk = LOCAL_UNKNOWN_CODE_ID[tech_scope]
        remapped = np.empty_like(tech_codes)
        n_outside = 0
        for i, c in enumerate(tech_codes):
            tv = CODE_TO_TECH_VARIANT.get(int(c))
            if tv is None or tv[0] != tech_scope:
                remapped[i] = unk
                n_outside += 1
            else:
                remapped[i] = local_table.get(tv, unk)
        tech_codes = remapped
        if n_outside:
            print(f"  tech_scope={tech_scope}: {n_outside} rows fell to "
                  f"UNKNOWN (expected 0 if exclude_techs is set correctly)")
        print(f"  tech_scope={tech_scope}: remapped to local vocab "
              f"(size={len(local_table) + 1})")

    if output_subset is not None:
        for c in output_subset:
            if c not in declared_columns:
                raise ValueError(
                    f"output_subset column {c!r} not in dataset columns")
        col_idx = [declared_columns.index(c) for c in output_subset]
        outputs = outputs[:, col_idx]
        print(f"  Output subset: kept {len(output_subset)} cols "
              f"{output_subset}")

    if split_mode == "combo":
        combo_strata = np.column_stack([tech_codes, geometry[:, :3]])
        train_idx, val_idx, test_idx = grouped_split_indices(
            combo_strata, train_ratio, val_ratio, seed,
        )
    elif split_mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(outputs))
        n_train = int(len(perm) * train_ratio)
        n_val = int(len(perm) * val_ratio)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]
    else:
        raise ValueError("split_mode must be 'combo' or 'random'")

    normalizer = normalizer_for(norm_mode)
    selected_columns = (
        list(output_subset) if output_subset is not None
        else declared_columns
    )
    normalizer.fit(
        inputs[train_idx], geometry[train_idx], outputs[train_idx],
        output_columns=selected_columns,
    )

    def _make(idxs: np.ndarray) -> MOSFETDataset:
        x = normalizer.normalize_inputs(inputs[idxs], geometry[idxs])
        y = normalizer.normalize_outputs(outputs[idxs])
        return MOSFETDataset(
            x, y, tech_codes[idxs],
            sample_class=sample_class[idxs],
            sample_class_names=sample_class_names)

    train_ds = _make(train_idx)
    val_ds = _make(val_idx)
    test_ds = _make(test_idx)

    print(f"  Split ({split_mode}): train={len(train_ds)} "
          f"val={len(val_ds)} test={len(test_ds)}")
    return train_ds, val_ds, test_ds, normalizer
