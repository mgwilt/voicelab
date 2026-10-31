"""CAM++ speaker conditioning matching the pinned official Ming frontend."""

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from torchaudio.compliance.kaldi import fbank

from voicelab_models.ming_tts.model import MODEL_ID, MODEL_REVISION


class SpeakerEncoder:
    def __init__(self) -> None:
        path = hf_hub_download(MODEL_ID, "campplus.onnx", revision=MODEL_REVISION)
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def encode(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        if audio.size < 400:
            raise ValueError("Speaker reference must contain at least 25 ms of speech")
        features = fbank(
            torch.from_numpy(audio.copy())[None, :],
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        features -= features.mean(dim=0, keepdim=True)
        values: object = self.session.run(
            None, {self.session.get_inputs()[0].name: features[None].numpy()}
        )
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError("CAM++ must return one speaker embedding")
        embedding = values[0]
        if (
            not isinstance(embedding, np.ndarray)
            or embedding.size != 192
            or not np.isfinite(embedding).all()
        ):
            raise ValueError("CAM++ returned an invalid speaker embedding")
        return np.asarray(embedding.reshape(192), dtype=np.float32)
