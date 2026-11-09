# Contributing to VAR-Offside

Thanks for helping. Bug reports, wrong calls, docs fixes and code are all welcome.

## Before you start

- **Small fixes** (typos, docs, an obvious bug): open a pull request directly.
- **Anything bigger** (a feature, a new calibration method, a change to the offside logic): open an issue first so we can agree on the approach before you spend time on it.
- **Security problems**: don't open a public issue. See [SECURITY.md](SECURITY.md).

## Set up

```bash
git clone https://github.com/<you>/var-offside.git
cd var-offside
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest
```

The tests need no model weights or GPU. To run the app itself, download the models as described in [docs/getting-started/models.md](docs/getting-started/models.md).

## Making a change

1. Create a branch from `main`: `git checkout -b fix/short-description`.
2. Write the change and its tests. Every fix to the offside logic needs a test in `tests/web/test_offside_rules.py` that fails without it.
3. Run `pytest` and `ruff check .`.
4. Update the docs in `docs/` if you change behaviour, a setting (also update `config.example.yaml`), the CLI, or the web API (`docs/reference/web-api.md`).
5. Add a line under **Unreleased** in [CHANGELOG.md](CHANGELOG.md).
6. Open a pull request and fill in the template.

### Commit messages

Write the subject in the imperative, under about 70 characters: `Fix keeper choice when two goalkeepers are detected`. Explain the why in the body when it isn't obvious.

### Code style

- Lint with `ruff`. `pre-commit` runs it for you.
- Use `logging`, not `print`, in new library code.
- Build file paths with `var_offside.config.paths`, never from the current directory.
- Read settings from `get_settings()`; add new ones to `var_offside/config/settings.py` with a default.
- Keep anything that opens an OpenGL context on the render thread (see [Architecture](docs/development/architecture.md)).

## What not to commit

- **Model weights** (`*.pt`, `*.ckpt`, PnLCalib weights). They go in `weights/`, which is ignored.
- **Broadcast footage or screenshots.** Use synthetic images or frames you have the rights to share. Rights-cleared images go in `docs/assets/`.
- **Secrets.** `var_offside/.env` and `config.yaml` are ignored; keep it that way.

## Reporting a wrong call

Use the **Wrong call** issue template. Include the verdict you got, the one you expected and why, the calibration status, and either the frame (only if you may share it) or the exported `report.json`.

## Licence

By contributing, you agree that your contributions are licensed under [AGPL-3.0-or-later](LICENSE), the project's licence. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).
