import json
from datetime import datetime
from pathlib import Path


class Transcript:
    def __init__(self, storage_dir, session_id):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(storage_dir) / "transcripts" / f"{stamp}_{session_id}.md"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"# DDx Transcript\n\nsession: `{session_id}`\nstarted: `{stamp}`\n\n")

    def append_turn(self, user_message, result):
        totals = result["llm_totals"]
        lines = [
            f"## Turn",
            "",
            f"Patient: {user_message}",
            "",
            f"Assistant: {result['question']}",
            "",
            f"LLM cumulative: calls={len(result['llm_calls'])} input={totals['input_tokens']} cached_input={totals['cached_input_tokens']} output={totals['output_tokens']} total={totals['total_tokens']} cost=${totals['cost_usd']:.6f}",
            "",
            "### LLM Calls",
            "",
        ]
        for call in result["llm_calls"]:
            lines.extend(
                [
                    f"#### Call {call['call']}",
                    "",
                    "[INPUT]",
                    "",
                    "```text",
                    f"system: {call.get('system', '')}",
                    f"user: {call.get('user', '')}",
                    "```",
                    "",
                    "[OUTPUT]",
                    "",
                    "```text",
                    call.get("output", ""),
                    "```",
                    "",
                    f"[TOKENS] input={call['input_tokens']} cached_input={call['cached_input_tokens']} output={call['output_tokens']} total={call['total_tokens']} cost=${call['cost_usd']:.6f}",
                    "",
                ]
            )
        lines.extend(
            [
                "### Form",
                "",
                "```json",
                json.dumps(result["form"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        with self.path.open("a") as f:
            f.write("\n".join(lines) + "\n")
