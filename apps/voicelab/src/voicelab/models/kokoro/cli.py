"""Speech synthesis with retained audio and measurements."""

import argparse

from voicelab.runs.synthesis.cli import RunArguments, add_run_arguments, run
from voicelab.runs.synthesis.execution import ModelEvidence
from voicelab_models.kokoro.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    Request,
)


class Arguments(RunArguments):
    voice: str
    speed: float
    lang_code: str | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="kokoro synthesis and RTF measurements"
    )
    add_run_arguments(parser)
    parser.add_argument("--voice", type=str, default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--lang-code", choices=("a", "b", "e", "f", "h", "i", "j", "p", "z")
    )
    args = parser.parse_args(namespace=Arguments())
    request = Request.model_validate(
        {
            "text": args.text,
            "voice": args.voice,
            "speed": args.speed,
            "lang_code": args.lang_code,
        }
    )
    from voicelab_inference.kokoro.mlx.runtime import KokoroMLX

    runtime = KokoroMLX()
    evidence = ModelEvidence(
        MODEL_ID,
        MODEL_REVISION,
        CHECKPOINT_ID,
        CHECKPOINT_REVISION,
        SAMPLE_RATE,
        (
            "mlx",
            "mlx-audio",
            "numpy",
            "phonemizer",
            "espeakng-loader",
            "misaki",
            "spacy",
            "en-core-web-sm",
            "pyopenjtalk",
            "fugashi",
            "unidic",
            "pypinyin",
            "jieba",
        ),
    )
    run(parser, args, request, runtime, evidence)
