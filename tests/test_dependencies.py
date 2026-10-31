"""Enforce module boundaries without one distribution per model or backend."""

import ast
import sys
import unittest
from pathlib import Path


class DependencyTests(unittest.TestCase):
    def test_package_import_boundaries(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for distribution in (root / "packages").iterdir():
            source_root = distribution / "src"
            for source in source_root.rglob("*.py"):
                parts = source.relative_to(source_root).with_suffix("").parts
                namespace = parts[0]
                allowed = ["voicelab_core"]
                if namespace == "voicelab_models":
                    allowed.append(".".join(parts[:2]))
                elif namespace == "voicelab_inference":
                    allowed.extend(
                        [
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
                            if namespace == "voicelab_models":
                                self.assertNotIn(
                                    module, {"torch", "transformers", "mlx", "numpy"}
                                )
                            if namespace == "voicelab_core":
                                self.assertIn(
                                    module, sys.stdlib_module_names | {"voicelab_core"}
                                )
