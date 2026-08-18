import json
import unittest
from pathlib import Path

from anylabeling.views.labeling.utils.auto_labeling_contracts import (
    canonical_document_digest_v1,
    raw_file_sha256_v1,
    semantic_annotation_digest_v1,
)


class Phase1DigestGoldenVectorTests(unittest.TestCase):
    def test_frozen_document_semantic_and_raw_digests(self):
        vector_path = Path(__file__).with_name(
            "auto_labeling_phase1_golden_v1.json"
        )
        vector = json.loads(vector_path.read_text(encoding="utf-8"))
        expected = vector["expected"]
        self.assertEqual(
            canonical_document_digest_v1(vector["document"]),
            expected["document_digest"],
        )
        self.assertEqual(
            semantic_annotation_digest_v1(vector["document"]),
            expected["semantic_digest"],
        )
        self.assertEqual(
            raw_file_sha256_v1(bytes.fromhex(vector["raw_hex"])),
            expected["raw_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
