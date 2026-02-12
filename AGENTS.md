# Repository Guidelines

## Project Structure & Module Organization
Core training/inference code is in:
- `model/` (model definitions like DMD, GAN, SID, ODE)
- `pipeline/` (causal/bidirectional inference and training pipelines)
- `trainer/` (trainer implementations selected by config)
- `wan/` (Wan backbone, attention modules, configs, distributed helpers)
- `utils/` and `demo_utils/` (datasets, schedulers, wrappers, memory helpers)

Entry points are `train.py`, `inference.py`, and `demo.py`. Configs live in `configs/` (base defaults + per-experiment YAMLs), prompts in `prompts/`, and helper data scripts in `scripts/`.

## Build, Test, and Development Commands
- `uv sync --extra dev` installs runtime + dev dependencies from `pyproject.toml`.
- `pip install -r requirements.txt && python setup.py develop` matches the README setup flow.
- `bash DS_PC_inference.sh` runs Deep Sink + Participative Compression inference.
- `bash DS_inference.sh` runs Deep Sink-only inference.
- `python train.py --config_path configs/self_forcing_dmd.yaml --logdir ./logs` starts training.

If you are validating a change, run a quick smoke inference with a single prompt file (for example `prompts/MovieGenVideoBench_txt/line_0010.txt`).

## Coding Style & Naming Conventions
Target Python is 3.10. Use 4-space indentation and PEP 8 defaults.
- Functions/variables/files: `snake_case`
- Classes: `PascalCase`
- Config files: descriptive snake-case style (for example `self_forcing_dmd_sink14.yaml`)

Use `black` and `isort` from the `dev` extras before opening a PR:
- `uv run black .`
- `uv run isort .`

## Testing Guidelines
There is currently no dedicated `tests/` suite committed. For now:
- add focused unit tests with `pytest` when introducing new logic
- name tests `test_<module>.py`
- run `uv run pytest` for any added tests
- include at least one inference/training smoke check in PR notes

## Commit & Pull Request Guidelines
Recent history uses short, imperative messages (for example `feat: add uv`, `code release`). Follow that pattern:
- prefer `type: short summary` (`feat`, `fix`, `refactor`, `docs`)
- keep subject concise and scoped

PRs should include:
- what changed and why
- exact config/command used for validation
- sample artifact paths (for example output MP4 path) when behavior changes
- linked issue or experiment context when applicable
