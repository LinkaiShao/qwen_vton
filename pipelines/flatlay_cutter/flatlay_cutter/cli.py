"""Portable automatic setup and frozen-model cutting commands."""
import argparse
import json
import re
import time
from pathlib import Path

from .io import atomic, digest, export_cutout, read_image


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("IDs must contain only letters, numbers, dots, underscores and hyphens")
    return value


def load_manifest(path, data_root=None):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    root = Path(data_root).resolve() if data_root else path.parent
    if not isinstance(data.get("skus"), list) or not data["skus"]:
        raise ValueError("Manifest requires a nonempty skus list")
    keys = set()
    for sku in data["skus"]:
        key = safe_id(sku["key"])
        if key in keys:
            raise ValueError("Duplicate SKU key: " + key)
        keys.add(key)
        if "views" in sku and ("references" in sku or "targets" in sku):
            raise ValueError("Use views OR references/targets, not both, for " + key)
        groups = [sku["views"]] if "views" in sku else [sku.get("references", []), sku.get("targets", [])]
        ids = set()
        for group in groups:
            if not isinstance(group, list):
                raise ValueError("views/references/targets must be lists")
            for view in group:
                vid = safe_id(view["id"])
                if vid in ids:
                    raise ValueError("Repeated image ID within SKU " + key + ": " + vid)
                ids.add(vid)
                for field in ("source", "mask"):
                    if field == "mask" and not view.get(field):
                        continue
                    source = Path(view[field]).expanduser()
                    source = (root / source).resolve() if not source.is_absolute() else source.resolve()
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    view[field] = str(source)
                    expected = view.get("source_sha256" if field == "source" else "mask_sha256")
                    if expected and digest(source) != expected:
                        raise ValueError("Input hash mismatch: " + str(source))
        references = {r["source"] for r in sku.get("references", [])}
        if any(t["source"] in references for t in sku.get("targets", [])):
            raise ValueError("A worn target cannot also be a product-only reference: " + key)
    return data


def code_hashes():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def cut_manifest(manifest, output, local_files_only=False):
    from collections import Counter
    import numpy as np
    from PIL import Image
    from .base import SAM_MODEL, SAM_REVISION
    from .engine import ReferenceCutter
    from .setup import GENERIC

    output = Path(output)
    hashes = code_hashes()
    model = None
    records = []
    for sku in manifest["skus"]:
        if sku.get("status", "ready") != "ready" or not sku.get("references") or not sku.get("targets"):
            records.append({"sku": sku["key"], "status": sku.get("status", "skipped_missing_inputs"),
                            "error": sku.get("error")})
            continue
        try:
            for ref in sku["references"]:
                queries = ref.get("reference_queries")
                if (not isinstance(queries, list) or not queries or
                        any(not isinstance(q, str) or not q.strip() or q.lower().strip() in GENERIC for q in queries)):
                    raise ValueError("Run prepare first: specific reference_queries required for " + ref["id"])
            if model is None:
                model = ReferenceCutter(local_files_only=local_files_only)
            setup_seconds = model.prepare(sku)
        except Exception as exc:
            records.append({"sku": sku["key"], "status": "reference_error", "error": str(exc)})
            print("REFERENCE_ERROR", sku["key"], str(exc), flush=True)
            continue
        for target in sku["targets"]:
            folder = output / "outputs" / sku["key"] / target["id"]
            row = {"sku": sku["key"], "target": target["id"], "source": target["source"],
                   "source_sha256": digest(target["source"]), "code_sha256": hashes,
                   "reference_setup_seconds": setup_seconds}
            try:
                started = time.perf_counter()
                image = read_image(target["source"])
                mask, evidence = model.cut(image, sku["key"])
                row.update(evidence, status="cut" if mask.any() else "empty_mask")
                folder.mkdir(parents=True, exist_ok=True)
                if mask.any():
                    row.update(export_cutout(image, mask, folder))
                else:
                    Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "mask.png")
                row["file_to_export_seconds"] = time.perf_counter() - started
            except Exception as exc:
                row.update(status="processing_error", error=str(exc))
            atomic(folder / "result.json", row)
            records.append(row)
            print("CUT", sku["key"], target["id"], row["status"], flush=True)
        # Only this SKU's reference banks are needed; text embeddings remain cached.
        del model.refs[sku["key"]]
        atomic(output / "PROGRESS.json", {"records": len(records), "last_sku": sku["key"]})
    summary = {"records": records, "counts": dict(Counter(r["status"] for r in records)),
               "model": SAM_MODEL, "revision": SAM_REVISION, "code_sha256": hashes,
               "training": False, "per_target_vlm_calls_in_cut": 0}
    atomic(output / "RESULTS.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cut worn garments using product-only image references")
    parser.add_argument("phase", choices=("prepare", "cut", "run"),
                        help="prepare: automatic photo selection/naming; cut: SAM3 only; run: both")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New or empty output directory; existing results are never overwritten")
    parser.add_argument("--data-root", type=Path, help="Base for relative input paths; default: manifest directory")
    parser.add_argument("--offline", action="store_true", help="Use only cached Hugging Face model files")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest, args.data_root)
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty; choose a fresh directory: " + str(output))
        output.mkdir(parents=True, exist_ok=True)
        if args.phase in {"prepare", "run"}:
            from .setup import prepare_manifest
            manifest = prepare_manifest(manifest, output / "setup_cache", args.offline)
            atomic(output / "prepared_manifest.json", manifest)
        if args.phase in {"cut", "run"}:
            if any("views" in s for s in manifest["skus"]):
                raise ValueError("cut requires references/targets; use prepare or run for raw views")
            summary = cut_manifest(manifest, output, args.offline)
            print(json.dumps(summary["counts"], sort_keys=True))
            return 2 if any(r["status"] in {"setup_error", "reference_error", "processing_error", "empty_mask"}
                            for r in summary["records"]) else 0
        return 2 if any(s["status"] == "setup_error" for s in manifest["skus"]) else 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
