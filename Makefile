.PHONY: sync check format check-web check-sesame check-breeze

sync:
	uv sync --all-packages --extra mlx --locked

check:
	$(MAKE) check-web
	uv run --all-packages --extra mlx --locked ruff check .
	uv run --all-packages --extra mlx --locked ruff format --check .
	uv run --all-packages --extra mlx --locked ty check
	uv run --all-packages --extra mlx --locked python -m unittest discover -s tests -v
	$(MAKE) check-sesame
	$(MAKE) check-breeze

check-web:
	bun run --cwd apps/voicelab-web check
	bun run --cwd apps/voicelab-web build
	bun run --cwd apps/voicelab-web test

check-sesame:
	uv run --all-packages --extra mlx --locked python -m unittest discover -s packages/voicelab-inference/tests/sesame -v

format:
	uv run --all-packages --extra mlx --locked ruff format .

check-breeze:
	uv run --all-packages --extra mlx --locked python -m unittest discover -s packages/voicelab-inference/tests/breeze -v
