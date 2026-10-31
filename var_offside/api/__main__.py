"""Run the FastAPI service with uvicorn."""

import argparse

import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Run the VAR Offside API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload")
    return parser.parse_args()


def main():
    args = parse_args()
    uvicorn.run(
        "var_offside.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
