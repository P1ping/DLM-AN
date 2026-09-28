"""Guard against accidentally publishing development-only code or branding."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseSurfaceTests(unittest.TestCase):
    def test_no_contextual_dlm_or_scratch_scripts(self):
        self.assertFalse(list(ROOT.rglob("dlm" + "_ctx*.py")))
        self.assertFalse(list(ROOT.glob("tmp_*.sh")))
        self.assertFalse(list((ROOT / "configs/training").glob("*")))
        self.assertFalse((ROOT / ("flex" + "_an")).exists())

    def test_no_private_namespace_in_public_sources(self):
        for path in [ROOT / "README.md", *ROOT.rglob("*.py"), *ROOT.rglob("*.yaml")]:
            if "__pycache__" in path.parts:
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                text = path.read_text(encoding="utf-8").lower()
                self.assertNotIn("flex" + "_an", text)
                self.assertNotIn("flex" + "an", text)

    def test_dlm_has_no_reference_token_conditioning(self):
        source = (ROOT / "dlm_an/model/dlm/dlm.py").read_text(encoding="utf-8")
        for name in (
            "ref" + "_src_tokens",
            "ref" + "_tgt_tokens",
            "ctx" + "_cfg_scale",
            "sample" + "_ctx_lens",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
