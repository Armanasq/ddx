import json
from pathlib import Path


EMPTY_FORM = {
    "wf": "ddx",
    "session_id": "",
    "cc": "",
    "frame": {},
    "hypotheses": [],
    "facts": {},
    "conversation": [],
    "asked": [],
    "retrieved": [],
}


class Store:
    def __init__(self, directory, session_id):
        self.path = Path(directory) / f"{session_id}.json"

    def init(self):
        form = json.loads(json.dumps(EMPTY_FORM))
        form["session_id"] = self.path.stem
        self.save(form)
        return form

    def load(self):
        if not self.path.exists():
            return self.init()
        form = json.loads(self.path.read_text())
        return normalize_form(form, self.path.stem)

    def save(self, form):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(form, ensure_ascii=False, indent=2))


def merge_frame(form, result):
    """Apply frame_prompt result into the form (first turn)."""
    form = normalize_form(form, form.get("session_id", "session"))
    if result.get("cc"):
        form["cc"] = result["cc"]
    for key, value in (result.get("facts") or {}).items():
        v = _coerce(value)
        if v:
            form["facts"][key] = v
    if result.get("frame"):
        form["frame"] = result["frame"]
    if result.get("hypotheses"):
        form["hypotheses"] = result["hypotheses"]
    return form


def merge_update(form, result):
    """Apply hypothesis_update result into the form (subsequent turns)."""
    form = normalize_form(form, form.get("session_id", "session"))
    for key, value in (result.get("new_facts") or {}).items():
        v = _coerce(value)
        if v:
            form["facts"][key] = v
    if result.get("hypotheses"):
        form["hypotheses"] = result["hypotheses"]
    # Refresh the problem representation each turn so retrieval targets stay current.
    pr = (result.get("problem_representation") or "").strip()
    if pr:
        form.setdefault("frame", {})["one_liner"] = pr
    return form


def normalize_form(form, session_id):
    out = {
        "wf": "ddx",
        "session_id": form.get("session_id") or session_id,
        "cc": form.get("cc", ""),
        "frame": form.get("frame") or {},
        "hypotheses": form.get("hypotheses") or [],
        "facts": dict(form.get("facts") or {}),
        "conversation": form.get("conversation") or [],
        "asked": form.get("asked") or [],
        "retrieved": form.get("retrieved") or [],
    }
    if form.get("ddx"):
        out["ddx"] = form["ddx"]
    if not out["facts"]:
        for item in form.get("known") or []:
            text = str(item).strip()
            if text:
                out["facts"][f"legacy_{len(out['facts']) + 1}"] = text
    return out


def _coerce(value):
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()
