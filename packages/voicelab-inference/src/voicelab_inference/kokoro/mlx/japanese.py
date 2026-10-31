"""Initialize upstream Japanese pronunciation against the verified local cache."""

import threading
from pathlib import Path
from unittest.mock import patch

import unidic
from mlx_audio.tts.models.kokoro.kokoro import Model
from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

from voicelab_inference.kokoro.mlx.dictionary import dictionary_path

_INITIALIZATION = threading.Lock()


def install_pipeline(model: Model, path: Path) -> None:
    if "j" in model._pipelines:
        return
    with _INITIALIZATION:
        dictionary = dictionary_path()
        # Fugashi reads this package setting at Tagger construction. Restore it
        # after construction; the resident tagger owns its dictionary handle.
        with patch.object(unidic, "DICDIR", str(dictionary)):
            pipeline = KokoroPipeline(
                lang_code="j",
                model=model,
                repo_id=model.REPO_ID,
                model_path=path,
            )
        model._pipelines["j"] = pipeline
