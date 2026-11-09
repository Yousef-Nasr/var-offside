"""Run the browser-based VAR offside server.

Usage::

    var-offside-web [--config config.yaml] [--host 127.0.0.1] [--port 8765] [--no-browser]
    python -m var_offside.web ...

Defaults come from the ``server`` section of ``config.yaml``; flags win. The
browser opens automatically after a short delay unless ``--no-browser`` is
passed or ``server.open_browser`` is false.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import webbrowser


def _open_browser(url: str, delay: float = 1.0) -> None:
    """Open ``url`` in the default browser after ``delay`` seconds."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - opening a browser is best effort
        logging.getLogger(__name__).warning("Could not open a browser for %s", url)


def _preload_on_render_thread() -> None:
    """Do the body model's OpenGL-sensitive import early, so the first view is quick."""
    from var_offside.web.live3d import preload_body_model_imports
    from var_offside.web.sessions import RENDER_THREAD

    RENDER_THREAD.submit(preload_body_model_imports)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and start uvicorn."""
    parser = argparse.ArgumentParser(
        prog="var-offside-web",
        description="Serve the VAR offside web app.",
    )
    parser.add_argument(
        "--config",
        help="Path to a config.yaml (default: config.yaml in the project folder)",
    )
    parser.add_argument("--host", help="Bind host (default from config: 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Bind port (default from config: 8765)")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser on startup",
    )
    args = parser.parse_args(argv)

    if args.config:
        # Set before anything reads the settings, so every module sees it.
        os.environ["VAR_OFFSIDE_CONFIG"] = os.path.abspath(args.config)

    from var_offside.config import get_settings

    settings = get_settings()
    host = args.host or settings.server.host
    port = args.port or settings.server.port
    open_browser = settings.server.open_browser and not args.no_browser

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    threading.Thread(target=_preload_on_render_thread, daemon=True).start()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise SystemExit(
            "uvicorn is required to run the web app (pip install uvicorn)"
        ) from exc

    if settings.config_file:
        logging.getLogger(__name__).info("Using config file %s", settings.config_file)

    browser_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{browser_host}:{port}/"

    if open_browser:
        threading.Thread(
            target=_open_browser,
            args=(url,),
            name="open-browser",
            daemon=True,
        ).start()

    uvicorn.run("var_offside.web.server:app", host=host, port=port)


if __name__ == "__main__":
    main()
