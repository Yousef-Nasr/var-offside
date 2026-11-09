# Security policy

## Supported versions

While the project is before 1.0, only the latest release on `main` gets security fixes.

## Reporting a vulnerability

Please **don't open a public issue**. Report it privately through GitHub: **Security → Report a vulnerability** on this repository ([direct link](https://github.com/Yousef-Nasr/var-offside/security/advisories/new)).

Include what you found, how to reproduce it, and what an attacker could do with it. You'll get an answer within 7 days. Once a fix is released, you'll be credited in the advisory unless you'd rather not be.

## Things to know when running VAR-Offside

- **The web server has no login.** It binds to `127.0.0.1` by default. Setting `--host 0.0.0.0` lets anyone on your network upload files and use your GPU; only do that on a network you trust, or behind an authenticating reverse proxy.
- **Model files are code.** PyTorch checkpoints can run code when loaded. Only use weights from the sources listed in [docs/getting-started/models.md](docs/getting-started/models.md).
- **Auto-review sends frames to a third party.** With an OpenRouter key set and Auto-review switched on, the frame and player crops go to OpenRouter and the model provider. It's off by default.
- **Secrets** belong in `var_offside/.env` or the environment, never in `config.yaml` or a commit.
