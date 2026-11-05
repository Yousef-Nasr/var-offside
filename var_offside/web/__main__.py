"""Run the browser-based VAR offside server.

Usage::

    python -m var_offside.web [--host 127.0.0.1] [--port 8765] [--no-browser]

The browser opens automatically after a short delay unless ``--no-browser`` is
passed.
"""

from __future__ import annotations

import argparse
import logging
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
        prog="python -m var_offside.web",
        description="Serve the VAR offside web app.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser on startup",
    )
    args = parser.parse_args(argv)

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

    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{browser_host}:{args.port}/"

    if not args.no_browser:
        threading.Thread(
            target=_open_browser,
            args=(url,),
            name="open-browser",
            daemon=True,
        ).start()

    uvicorn.run("var_offside.web.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
