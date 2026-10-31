"""Launcher routing and reproducible setup without loading inference packages."""

import io
import tempfile
import unittest
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from voicelab.launch.cli import (
    Arguments,
    commands,
    main,
    parser_for,
    run_arguments,
)
from voicelab.launch.workspace import Workspace


class FakeRunner:
    def __init__(self, *, fail_build: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fail_build = fail_build

    def run(self, command: Sequence[str], cwd: Path) -> None:
        self.calls.append(tuple(command))
        if tuple(command) == ("bun", "run", "build"):
            if self.fail_build:
                raise RuntimeError("Build failed")
            output = cwd / "src/voicelab_web/static/dist"
            output.mkdir(parents=True, exist_ok=True)
            for name in ("main.js", "capture-worklet.js"):
                (output / name).write_text("built")


def fixture(root: Path) -> Path:
    web = root / "apps/voicelab-web"
    (web / "frontend").mkdir(parents=True)
    (web / "types").mkdir()
    for name in ("package.json", "bun.lock", "tsconfig.json", "bunfig.toml"):
        (web / name).write_text("fixture")
    (web / "frontend/main.ts").write_text("initial")
    return web


class LauncherTests(unittest.TestCase):
    def test_help_and_models_do_not_prepare_workspace(self) -> None:
        with (
            patch.object(Workspace, "prepare_python") as prepare,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(["models"]), 0)
            with self.assertRaises(SystemExit) as result:
                main(["--help"])
            self.assertEqual(result.exception.code, 0)
        prepare.assert_not_called()

    def test_runtime_options_do_not_consume_model_arguments(self) -> None:
        parser = parser_for(commands())
        args, extra = parser.parse_known_args(
            ["run", "breeze", "--text", "Hello", "--runtime", "mlx"],
            namespace=Arguments(),
        )
        self.assertEqual(args.runtime, "mlx")
        self.assertEqual(extra, ["--text", "Hello"])
        for model_name, runtime_name, executable in (
            ("qwen-omni", "mlx", "voicelab-qwen-omni"),
            ("minicpm", "llama.cpp", "voicelab-minicpm"),
        ):
            with self.subTest(model=model_name):
                selected, options = parser.parse_known_args(
                    ["run", model_name, "--input", "question.wav"],
                    namespace=Arguments(),
                )
                self.assertEqual(selected.runtime, runtime_name)
                self.assertEqual(
                    selected.model_command.runtimes[0].executable, executable
                )
                self.assertEqual(options, ["--input", "question.wav"])
        with (
            redirect_stdout(io.StringIO()),
            patch("sys.stderr", new=io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parser.parse_args(["run", "breeze", "--runtime", "cuda"])

    def test_default_outputs_are_fresh_but_explicit_output_is_preserved(self) -> None:
        model = commands()[2]
        first = run_arguments(model, model.runtimes[0], ["--text", "Hello"])
        second = run_arguments(model, model.runtimes[0], ["--text", "Hello"])
        self.assertNotEqual(first[-1], second[-1])
        explicit = ["--text", "Hello", "--output=artifacts/chosen"]
        self.assertEqual(run_arguments(model, model.runtimes[0], explicit), explicit)

    def test_rebuilds_only_for_changed_or_missing_assets(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("shutil.which", return_value="bun"),
        ):
            root = Path(directory)
            web = fixture(root)
            runner = FakeRunner()
            workspace = Workspace(root, runner)
            workspace.prepare_frontend()
            workspace.prepare_frontend()
            self.assertEqual(runner.calls.count(("bun", "run", "build")), 1)
            (web / "frontend/main.ts").write_text("changed")
            workspace.prepare_frontend()
            self.assertEqual(runner.calls.count(("bun", "run", "build")), 2)
            (web / "src/voicelab_web/static/dist/main.js").unlink()
            workspace.prepare_frontend()
            self.assertEqual(runner.calls.count(("bun", "run", "build")), 3)

    def test_failed_build_is_not_cached(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("shutil.which", return_value="bun"),
        ):
            root = Path(directory)
            fixture(root)
            with self.assertRaisesRegex(RuntimeError, "Build failed"):
                Workspace(root, FakeRunner(fail_build=True)).prepare_frontend()
            self.assertFalse((root / ".local/frontend-build.sha256").exists())
