# Contributing

Contributions are welcome: bug reports, wrong calls with the frame attached, docs fixes and code. The full guide is [CONTRIBUTING.md](https://github.com/Yousef-Nasr/var-offside/blob/main/CONTRIBUTING.md) in the repository. In short:

1. Open an issue first for anything bigger than a small fix, so we can agree on the approach.
2. Fork, then set up:

    ```bash
    git clone https://github.com/<you>/var-offside.git
    cd var-offside
    pip install -e ".[dev]"
    pre-commit install
    ```

3. Make the change with tests. Run `pytest` and `ruff check .`.
4. Update the docs in `docs/` when behaviour, settings or the API change.
5. Open a pull request and fill in the template.

**Never commit model weights or broadcast footage.** Use synthetic data or frames you have the rights to share, and add them with `git add -f` only into `docs/assets/`.

## Good places to start

- A new calibration method in `var_offside/calibration/` (subclass `BaseCalibration`, register it in `factory.py`).
- Better messages when a stage can't finish.
- Tests for code paths that have none, such as `teams/`.
- Translations of the web page.

See [Architecture](architecture.md) before changing the web server or the 3D view.

By contributing you agree that your work is licensed under the project's [AGPL-3.0 licence](../about/licensing.md) and that you'll follow the [Code of Conduct](https://github.com/Yousef-Nasr/var-offside/blob/main/CODE_OF_CONDUCT.md).
