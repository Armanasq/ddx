import json
import os
import time
from pathlib import Path
from threading import Lock


FIELDS = (
    "calls",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
    "input_cost_usd",
    "cached_cost_usd",
    "output_cost_usd",
    "cost_usd",
)


def _empty():
    return {f: 0 if "cost" not in f else 0.0 for f in FIELDS}


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class CostLedger:
    """Persistent, incrementally-updated cost ledger.

    Two artifacts under the storage dir:
      - cost_ledger.json : rolling totals for all-time and per-session (the running
                           cumulative development cost).
      - cost_log.jsonl   : append-only audit log, one line per LLM call, with the
                           per-call token counts and the input/output cost breakdown.

    Updates are atomic (temp file + os.replace) so a crash mid-write cannot corrupt
    the running totals.
    """

    def __init__(self, storage_dir):
        self.dir = Path(storage_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.dir / "cost_ledger.json"
        self.log_path = self.dir / "cost_log.jsonl"
        self._lock = Lock()

    def record_call(self, call, session_id, model=""):
        """Fold one completed LLM call into the ledger and append it to the audit log."""
        with self._lock:
            ledger = self._load()
            self._add(ledger["all_time"], call)
            session = ledger["sessions"].setdefault(session_id, _empty())
            self._add(session, call)
            session["last_updated"] = _now()
            ledger["updated_at"] = _now()
            self._save(ledger)
            self._append_log(call, session_id, model)
            return ledger

    def totals(self):
        """Return the current all-time and per-session rolling totals."""
        return self._load()

    # ── internals ────────────────────────────────────────────────────────────

    def _load(self):
        if self.ledger_path.exists():
            try:
                data = json.loads(self.ledger_path.read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
        else:
            data = {}
        data.setdefault("all_time", _empty())
        data.setdefault("sessions", {})
        # backfill any newly added fields onto an older ledger
        for f in FIELDS:
            data["all_time"].setdefault(f, 0 if "cost" not in f else 0.0)
        return data

    def _add(self, bucket, call):
        bucket["calls"] = bucket.get("calls", 0) + 1
        for f in FIELDS:
            if f == "calls":
                continue
            bucket[f] = round(bucket.get(f, 0) + call.get(f, 0), 10)

    def _save(self, ledger):
        tmp = self.ledger_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=2))
        os.replace(tmp, self.ledger_path)

    def _append_log(self, call, session_id, model):
        entry = {
            "ts": _now(),
            "session": session_id,
            "model": model,
            "call": call.get("call"),
            "input_tokens": call.get("input_tokens", 0),
            "cached_input_tokens": call.get("cached_input_tokens", 0),
            "output_tokens": call.get("output_tokens", 0),
            "total_tokens": call.get("total_tokens", 0),
            "input_cost_usd": call.get("input_cost_usd", 0.0),
            "cached_cost_usd": call.get("cached_cost_usd", 0.0),
            "output_cost_usd": call.get("output_cost_usd", 0.0),
            "cost_usd": call.get("cost_usd", 0.0),
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
