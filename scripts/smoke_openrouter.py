"""Send one frame to OpenRouter and print the validated frame check.

This makes one real paid call and sends the image to OpenRouter and the model
provider. It reads OPENROUTER_API_KEY and OPENROUTER_MODEL_ID from the
environment or var_offside/.env.

    python scripts/smoke_openrouter.py path/to/frame.jpg [--model provider/model]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from var_offside.agent.config import load_config  # noqa: E402
from var_offside.agent.llm import LLMError, OpenRouterClient  # noqa: E402
from var_offside.agent.prompts import FRAME_CHECK_PROMPT, SYSTEM_PROMPT  # noqa: E402
from var_offside.agent.schemas import FrameCheck  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", help="Override OPENROUTER_MODEL_ID for this call")
    args = parser.parse_args()

    config = load_config()
    if args.model:
        config = dataclasses.replace(config, model=args.model)

    client = OpenRouterClient(config)
    start = time.perf_counter()
    try:
        answer = client.ask("frame", SYSTEM_PROMPT, FRAME_CHECK_PROMPT, FrameCheck, images=[args.image])
    except (LLMError, ValueError) as exc:
        print(f"FAILED: {exc}")
        print(f"model={config.model} calls={client.usage.calls} cost_usd={client.usage.cost_usd:.6f}")
        return 1

    print(answer.model_dump_json(indent=2))
    print(
        f"model={config.model} seconds={time.perf_counter() - start:.1f} "
        f"calls={client.usage.calls} cost_usd={client.usage.cost_usd:.6f} "
        f"unpriced_calls={client.usage.unpriced_calls}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
