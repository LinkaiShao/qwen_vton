import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from flatlay_cutter.cli import load_manifest, main
from flatlay_cutter.io import digest, export_cutout


class PortableContractTests(unittest.TestCase):
    def test_export_preserves_native_rgb_holes_and_disconnected_straps(self):
        rng = np.random.default_rng(14)
        rgb = rng.integers(0, 256, (50, 40, 3), dtype=np.uint8)
        mask = np.zeros((50, 40), dtype=bool)
        mask[12:37, 9:30] = True
        mask[18:24, 14:20] = False
        mask[2:11, 4] = True
        with tempfile.TemporaryDirectory() as tmp:
            record = export_cutout(Image.fromarray(rgb), mask, tmp)
            x0, y0, x1, y1 = record["box_xyxy"]
            cut = np.asarray(Image.open(Path(tmp) / "cutout.png"))
            local = mask[y0:y1, x0:x1]
            self.assertTrue(np.array_equal(cut[:, :, 3] > 0, local))
            self.assertTrue(np.array_equal(cut[:, :, :3][local], rgb[y0:y1, x0:x1][local]))
            self.assertTrue(np.array_equal(np.asarray(Image.open(Path(tmp) / "mask.png")) > 0, mask))

    def test_manifest_rejects_changed_reference_and_reference_as_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 20), "red").save(root / "reference.png")
            Image.new("RGB", (20, 20), "blue").save(root / "worn.png")
            ref = {"id": "r", "source": "reference.png", "source_sha256": digest(root / "reference.png")}
            data = {"skus": [{"key": "sku", "references": [ref],
                              "targets": [{"id": "t", "source": "worn.png"}]}]}
            path = root / "manifest.json"
            path.write_text(json.dumps(data))
            self.assertEqual(load_manifest(path)["skus"][0]["references"][0]["source"], str(root / "reference.png"))
            Image.new("RGB", (20, 20), "green").save(root / "reference.png")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_manifest(path)
            del ref["source_sha256"]
            data["skus"][0]["targets"][0]["source"] = "reference.png"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "cannot also be"):
                load_manifest(path)

    def test_existing_results_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 20), "red").save(root / "source.png")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"skus": [{"key": "sku", "views": [{"id": "v", "source": "source.png"}]}]}))
            out = root / "out"
            out.mkdir()
            previous = out / "RESULTS.json"
            previous.write_text("preserve this earlier result")
            with self.assertRaises(SystemExit) as error:
                main(["run", "--manifest", str(manifest), "--output", str(out)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(previous.read_text(), "preserve this earlier result")

    def test_output_ids_cannot_escape_the_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "manifest.json"
            path.write_text(json.dumps({"skus": [{"key": "../elsewhere", "references": [], "targets": []}]}))
            with self.assertRaisesRegex(ValueError, "IDs must"):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
