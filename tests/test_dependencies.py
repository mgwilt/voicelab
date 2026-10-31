"""Enforce module boundaries without one distribution per model or backend."""

import ast
import sys
import unittest
from pathlib import Path


class DependencyTests(unittest.TestCase):
    def test_application_import_boundaries(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source_root = root / "apps/voicelab/src/voicelab"
        for source in source_root.rglob("*.py"):
            parts = source.relative_to(source_root).with_suffix("").parts
            allowed = ["voicelab_core"]
            if parts[0] == "launch":
                allowed = ["voicelab.launch"]
            elif parts[0] == "runs":
                allowed.extend(["voicelab.runs", "voicelab_models.audio"])
            elif parts[0] == "conversations":
                allowed.extend(
                    [
                        "voicelab.conversations",
                        "voicelab.runs.worker",
                        "voicelab_models",
                        "voicelab_inference",
                    ]
                )
            elif parts[0] == "models" and len(parts) > 2:
                allowed.extend(
                    [
                        f"voicelab.models.{parts[1]}",
                        "voicelab.runs",
                        "voicelab.launch.devices",
                        f"voicelab_models.{parts[1]}",
                        "voicelab_models.audio",
                        f"voicelab_inference.{parts[1]}",
                        "voicelab_inference.audio",
                    ]
                )
            for node in ast.walk(ast.parse(source.read_text())):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0, f"Use explicit imports: {source}")
                    names = [node.module] if node.module is not None else []
                for name in names:
                    module = name.split(".")[0]
                    with self.subTest(source=source, module=name):
                        if module == "voicelab" or module.startswith("voicelab_"):
                            self.assertTrue(
                                any(
                                    name == prefix or name.startswith(prefix + ".")
                                    for prefix in allowed
                                ),
                                f"{name} crosses {'.'.join(parts)} ownership",
                            )
                        if parts[0] == "conversations":
                            self.assertNotIn(
                                module,
                                {
                                    "fastapi",
                                    "starlette",
                                    "uvicorn",
                                    "aiortc",
                                    "voicelab_web",
                                },
                            )
                        if parts[0] == "launch":
                            self.assertIn(
                                module, sys.stdlib_module_names | {"voicelab"}
                            )
                        elif parts[0] == "runs":
                            self.assertNotIn(module, {"torch", "transformers", "mlx"})

    def test_package_import_boundaries(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for distribution in (root / "packages").iterdir():
            source_root = distribution / "src"
            for source in source_root.rglob("*.py"):
                parts = source.relative_to(source_root).with_suffix("").parts
                namespace = parts[0]
                allowed = ["voicelab_core"]
                if namespace == "voicelab_models":
                    allowed.append("voicelab_models.audio")
                    allowed.append(".".join(parts[:2]))
                elif namespace == "voicelab_inference":
                    allowed.extend(
                        [
                            "voicelab_models.audio",
                            "voicelab_inference.audio",
                            f"voicelab_models.{parts[1]}",
                            ".".join(parts[:3]),
                        ]
                    )
                for node in ast.walk(ast.parse(source.read_text())):
                    names: list[str] = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        if node.level:
                            self.fail(
                                f"Use explicit imports for boundary checks: {source}"
                            )
                        names = [node.module] if node.module is not None else []
                    for name in names:
                        module = name.split(".")[0]
                        with self.subTest(source=source, module=name):
                            if module == "voicelab" or module.startswith("voicelab_"):
                                self.assertTrue(
                                    any(
                                        name == prefix or name.startswith(prefix + ".")
                                        for prefix in allowed
                                    ),
                                    f"{name} crosses {'.'.join(parts)} ownership",
                                )
                            self.assertNotEqual(
                                module,
                                "pipecat",
                                f"Keep Pipecat in the application: {source}",
                            )
                            if namespace == "voicelab_models":
                                self.assertNotIn(
                                    module,
                                    {
                                        "torch",
                                        "transformers",
                                        "mlx",
                                        "numpy",
                                        "pipecat",
                                    },
                                )
                            if namespace == "voicelab_core":
                                self.assertIn(
                                    module, sys.stdlib_module_names | {"voicelab_core"}
                                )
