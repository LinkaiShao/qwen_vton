"""Automatic photo selection and reference naming, outside the fast cutting loop."""
import gc
import hashlib
import json
from pathlib import Path

from .io import atomic, digest, parse_selection, read_image
from .prompts import FRAMING_PROMPT, NAMING_PROMPT, SELECTION_PROMPT

QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
GENERIC = {"clothing", "garment", "clothes", "apparel", "fashion item"}


def parse_object(raw):
    for index, char in enumerate(raw):
        if char == "{":
            try:
                obj, _ = json.JSONDecoder().raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("Model did not return a JSON object")


def specific_names(names):
    if not isinstance(names, list) or not all(isinstance(q, str) for q in names):
        raise ValueError("reference_queries must be a list of strings")
    return list(dict.fromkeys(q.strip() for q in names if q.strip() and q.strip().lower() not in GENERIC))


class SetupModel:
    def __init__(self, cache, local_files_only=False):
        self.cache = Path(cache)
        self.local_files_only = local_files_only
        self.net = self.proc = None

    def ask(self, sources, prompt, max_new_tokens=200):
        import torch
        identity = {"model": QWEN_MODEL, "revision": QWEN_REVISION,
                    "prompt": prompt, "sources": [digest(p) for p in sources],
                    "use_fast": False, "max_pixels": 512 * 512,
                    "max_new_tokens": max_new_tokens, "do_sample": False}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + ".json")
        if path.exists():
            return json.loads(path.read_text())
        if self.net is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("Automatic setup requires a CUDA GPU with BF16 support")
            torch.set_num_threads(4)
            torch.backends.cuda.enable_cudnn_sdp(False)
            self.proc = AutoProcessor.from_pretrained(
                QWEN_MODEL, revision=QWEN_REVISION, local_files_only=self.local_files_only,
                max_pixels=512 * 512, use_fast=False)
            self.proc.tokenizer.padding_side = "left"
            self.net = AutoModelForImageTextToText.from_pretrained(
                QWEN_MODEL, revision=QWEN_REVISION, local_files_only=self.local_files_only,
                dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"}).eval()
        content = [{"type": "image"} for _ in sources] + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                            tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[read_image(p) for p in sources],
                           return_tensors="pt").to("cuda")
        with torch.inference_mode():
            tokens = self.net.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        raw = self.proc.decode(tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        record = {"identity": identity, "raw": raw}
        atomic(path, record)
        return record

    def close(self):
        import torch
        self.net = self.proc = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def prepare_manifest(manifest, cache, local_files_only=False):
    model = SetupModel(cache, local_files_only)
    prepared = []
    try:
        for sku in manifest["skus"]:
            row = {"key": sku["key"], "references": [], "targets": [], "excluded": []}
            try:
                if "views" in sku:
                    worn = []
                    for view in sku["views"]:
                        evidence = model.ask([view["source"]], SELECTION_PROMPT)
                        selection = parse_selection(evidence["raw"])
                        selected = dict(view, selection=selection, selection_evidence=evidence)
                        if selection["kind"] == "product_only":
                            row["references"].append(dict(selected, reference_queries=selection["items"]))
                        elif selection["kind"] == "worn":
                            worn.append(selected)
                        else:
                            row["excluded"].append(dict(selected, reason=selection["kind"]))
                    if row["references"]:
                        for view in worn:
                            evidence = model.ask([row["references"][0]["source"], view["source"]],
                                                 FRAMING_PROMPT, 32)
                            role = parse_object(evidence["raw"]).get("role")
                            if role not in {"overview", "detail"}:
                                raise ValueError("Invalid target framing classification: " + view["id"])
                            selected = dict(view, role=role, framing_evidence=evidence)
                            if role == "overview":
                                row["targets"].append(selected)
                            else:
                                row["excluded"].append(dict(selected, reason="detail"))
                    else:
                        row["excluded"].extend(dict(v, reason="no_product_only_reference") for v in worn)
                else:
                    # An upstream automatic selector can supply already-qualified inputs.
                    row["references"] = [dict(r) for r in sku.get("references", [])]
                    row["targets"] = [dict(t) for t in sku.get("targets", [])]
                for ref in row["references"]:
                    queries = specific_names(ref.get("reference_queries", []))
                    if not queries:
                        evidence = model.ask([ref["source"]], NAMING_PROMPT, 64)
                        queries = specific_names(parse_object(evidence["raw"]).get("queries"))
                        if not queries:
                            raise ValueError("No specific garment noun for reference " + ref["id"])
                        ref["naming_evidence"] = evidence
                    ref["reference_queries"] = queries
                    ref["source_sha256"] = digest(ref["source"])
                    if ref.get("mask"):
                        ref["mask_sha256"] = digest(ref["mask"])
                for target in row["targets"]:
                    target["source_sha256"] = digest(target["source"])
                row["status"] = ("ready" if row["targets"] and row["references"] else
                                 "no_product_only_reference" if not row["references"] else "no_worn_overview")
            except Exception as exc:
                row.update(status="setup_error", error=str(exc))
            prepared.append(row)
            print("PREPARE", row["key"], row["status"], flush=True)
    finally:
        model.close()
    return {"schema_version": 1, "phase": "prepared", "skus": prepared}
