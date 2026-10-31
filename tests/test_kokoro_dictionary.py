"""Verify downloaded pronunciation data before it enters the local cache."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voicelab_inference.kokoro.mlx.dictionary import dictionary_path


class DictionaryTests(unittest.TestCase):
    def test_rejects_corrupt_download_without_marking_cache_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict("os.environ", {"XDG_CACHE_HOME": directory}),
                patch("urllib.request.urlopen", return_value=io.BytesIO(b"corrupt")),
                self.assertRaisesRegex(ValueError, "checksum"),
            ):
                dictionary_path()
            self.assertFalse((Path(directory) / "voicelab/unidic-3.1.0").exists())
            self.assertEqual(list((Path(directory) / "voicelab").iterdir()), [])
