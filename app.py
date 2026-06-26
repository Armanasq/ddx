import argparse
import json

from ddx.runtime import Runtime
from ddx.transcript import Transcript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="session")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--llm-call", action="store_true")
    args = parser.parse_args()
    runtime = Runtime(session_id=args.session, trace_llm=args.llm_call, init_form=True)
    transcript = Transcript(runtime.config.storage_dir, args.session)
    print("ddx intake prototype")
    print("type exit to quit")
    print(f"transcript: {transcript.path}")
    while True:
        text = input("You: ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if not text:
            continue
        result = runtime.turn(text)
        transcript.append_turn(text, result)
        print("Assistant:", result["question"])
        sess = result.get("cost_session", {})
        allt = result.get("cost_all_time", {})
        print(
            "[COST] "
            f"turn=${result['llm_totals']['cost_usd']:.6f} "
            f"session=${sess.get('cost_usd', 0):.6f} ({sess.get('calls', 0)} calls) "
            f"all_time=${allt.get('cost_usd', 0):.6f} ({allt.get('calls', 0)} calls)"
        )
        if args.llm_call:
            totals = result["llm_totals"]
            print(
                "[LLM TURN] "
                f"calls={len(result['llm_calls'])} "
                f"input={totals['input_tokens']} "
                f"cached_input={totals['cached_input_tokens']} "
                f"output={totals['output_tokens']} "
                f"total={totals['total_tokens']} "
                f"cost=${totals['cost_usd']:.6f}"
            )
        if args.debug:
            print(json.dumps(result["form"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
