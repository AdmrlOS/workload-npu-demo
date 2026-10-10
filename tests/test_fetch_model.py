import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('fetch_model', Path(__file__).resolve().parents[1] / 'scripts/fetch-model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ModelStageTests(unittest.TestCase):
    def manifest(self, data):
        return {'name': 'test', 'model': {'path': '/models/model.gguf', 'source': 'https://example.com/model',
                'sha256': hashlib.sha256(data).hexdigest()}}

    def test_cached_pair_verified_and_manifest_written(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            payload = b'GGUFtest'
            (root / 'model.gguf').write_bytes(payload)
            manifest = self.manifest(payload)
            module.stage(manifest, root)
            self.assertEqual(json.loads((root / 'model.json').read_text()), manifest)

    def test_bad_existing_hash_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'model.gguf').write_bytes(b'GGUFother')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                module.stage(self.manifest(b'GGUFtest'), root)
            self.assertFalse((root / 'model.json').exists())

    def test_path_escape_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = self.manifest(b'GGUFtest')
            manifest['model']['path'] = '/models/../outside.gguf'
            with self.assertRaises(ValueError):
                module.stage(manifest, Path(d))

    def test_html_cannot_be_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'model.gguf').write_bytes(b'<html>')
            with self.assertRaisesRegex(ValueError, 'not a GGUF'):
                module.stage(self.manifest(b'<html>'), root)
