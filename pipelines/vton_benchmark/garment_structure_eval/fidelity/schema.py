"""Versioned JSONL contract. Proposals are never silently treated as labels."""
import hashlib
import json
import math
from pathlib import Path

CONCEPTS = ("logo_graphic", "button_placket", "pocket", "collar_neckline", "sleeve_cuff", "fabric_pattern")
VISIBILITY = ("visible", "absent", "occluded", "uncertain")
SPLITS = ("train", "calibration", "test", "development")
LABELS = ("faithful", "altered", "uncertain")


def split_for(garment_id):
    bucket = int(hashlib.sha256(garment_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 60 else "calibration" if bucket < 80 else "test"


def validate(rows):
    seen, groups = set(), {}
    for row in rows:
        key = row["id"]
        if key in seen:
            raise ValueError(f"Duplicate pair id: {key}")
        seen.add(key)
        if row.get("version") != 1 or row["split"] not in SPLITS:
            raise ValueError(f"Invalid version/split: {key}")
        gid = row["garment_id"]
        if groups.setdefault(gid, row["split"]) != row["split"]:
            raise ValueError(f"Garment split leakage: {gid}")
        if row.get("dataset_split") not in ("train", "test"):
            raise ValueError(f"Missing dataset provenance: {key}")
        if not row.get("target"):
            raise ValueError(f"Missing target: {key}")
        part_ids = set()
        for part in row["parts"]:
            if part["id"] in part_ids:
                raise ValueError(f"Duplicate part: {key}/{part['id']}")
            part_ids.add(part["id"])
            if part["concept"] not in CONCEPTS or part["visibility"] not in VISIBILITY:
                raise ValueError(f"Invalid concept/visibility: {key}")
            if part.get("label", "uncertain") not in LABELS:
                raise ValueError(f"Invalid fidelity label: {key}")
            box = part.get("box")
            if box is not None:
                if (len(box) != 4 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in box)
                        or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
                    raise ValueError(f"Invalid normalized xyxy box: {key}/{part['id']}")
            if part.get("reviewed") and not str(part.get("reviewer", "")).strip():
                raise ValueError(f"Reviewed annotation needs reviewer: {key}")
            if part.get("reviewed") and part["visibility"] == "visible" and box is None:
                raise ValueError(f"Visible annotation needs a target box: {key}")
    return rows


def usable(part):
    return (part.get("reviewed") is True and bool(part.get("reviewer", "").strip())
            and part["visibility"] == "visible" and part.get("owner") == "target_garment"
            and part.get("box") is not None)


def read_manifest(path):
    path = Path(path).resolve()
    rows = validate([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
    for row in rows:
        for field in ("target", "prediction", "garment", "parse"):
            if row.get(field):
                row[field] = str((path.parent / row[field]).resolve())
    return rows


def write_manifest(path, rows):
    validate(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
