import json

from ddx.prompts import (
    ddx_answer_prompt,
    frame_prompt,
    hypothesis_update_prompt,
    relevance_prompt,
    verification_prompt,
)


# ── LLM calls ────────────────────────────────────────────────────────────────

def build_frame(llm, user_message):
    """First turn: extract cc, facts, frame, ranked hypotheses (with acuity), and a proposed action."""
    return llm.complete_json_text(frame_prompt(user_message), max_tokens=800)


def update_hypotheses(llm, form, user_message):
    """Subsequent turns: reason from full conversation, update hypotheses, propose next action."""
    cc = form.get("cc", "")
    conversation = format_conversation(form.get("conversation", []))
    facts_text = format_facts(form.get("facts", {}))
    hypotheses_json = json.dumps(form.get("hypotheses", []), ensure_ascii=False)
    evidence_text = format_evidence(form.get("retrieved", []))
    prompt = hypothesis_update_prompt(cc, conversation, facts_text, hypotheses_json, evidence_text, user_message)
    return llm.complete_json_text(prompt, max_tokens=750)


def decomposed_queries(form):
    """The query decomposition a retrieved passage is judged against: the chief complaint
    plus each live (high/medium) diagnosis name. No keywords — names come from the LLM."""
    queries = [form.get("cc", "")]
    queries += [h.get("dx", "") for h in live_hypotheses(form) if h.get("dx")]
    return [q for q in queries if q]


def is_evidence_relevant(llm, patient_context, queries, chunk):
    """Low-cost relevance gate. Asks a cheap one-word (true/false) judgement whether the
    retrieved passage is relevant to this patient and the decomposed queries. No JSON,
    minimal tokens. Returns True only on an explicit 'true', so recall noise (off-topic or
    wrong-population passages) is dropped — without any keyword rules."""
    queries_text = "\n".join(f"- {q}" for q in queries)
    title = chunk.get("title", "")
    text = (chunk.get("text", "") or "")[:300]
    out = llm.complete_text_text(relevance_prompt(patient_context or "(no demographics given)", queries_text, title, text), max_tokens=10)
    return "true" in out.strip().lower()


def verify_candidates(llm, form):
    """Strict lane: check each live candidate against retrieved evidence, return verdicts."""
    problem = form.get("frame", {}).get("one_liner", "") or form.get("cc", "")
    candidates_text = format_candidates(live_hypotheses(form))
    evidence_text = format_evidence(form.get("retrieved", []))
    out = llm.complete_json_text(verification_prompt(problem, candidates_text, evidence_text), max_tokens=500)
    return out.get("verified", [])


def generate_ddx_answer(llm, form):
    """Final synthesis — every section built by Python from structured form data."""
    cc = form.get("cc", "")
    facts_text = format_facts(form.get("facts", {}))
    conversation = format_conversation(form.get("conversation", []))
    hypotheses_text = format_hypotheses(form.get("hypotheses", []))
    evidence_text = format_evidence(form.get("retrieved", []))
    prompt = ddx_answer_prompt(cc, facts_text, conversation, hypotheses_text, evidence_text)
    return llm.complete_text_text(prompt, max_tokens=1400)


# ── structural policy — reads ONLY frame properties, no disease keywords ──────

def live_hypotheses(form):
    """Plausible hypotheses (high/medium likelihood) — the focus of retrieval and the
    set that must be evidence-covered. A low-likelihood can't-miss diagnosis stays on the
    differential but only demands its own evidence once it becomes plausible, which prevents
    falsely marking it 'covered' by an unrelated earlier retrieval."""
    out = [h for h in form.get("hypotheses", []) if h.get("likelihood") in ("high", "medium")]
    return out or form.get("hypotheses", [])[:3]


def structural_retrieval_trigger(form):
    """True when the case frame is complex/high-risk enough that the next move
    cannot be justified from the frame alone. Pure structural read of LLM output."""
    frame = form.get("frame", {})
    systems = [s for s in frame.get("systems", []) if s]
    if len(systems) >= 2:                                              # multi-system
        return True
    if frame.get("time_course") == "acute" and frame.get("severity") == "high":  # acute + severe
        return True
    for h in form.get("hypotheses", []):                             # serious diagnosis live
        if h.get("likelihood") in ("high", "medium") and h.get("acuity") in ("emergent", "urgent"):
            return True
    return False


def needs_strict_verification(form):
    """Complex / high-risk cases get candidate-by-candidate verification; simple ones do not."""
    return structural_retrieval_trigger(form)


def has_attempted_retrieval(form):
    """Whether retrieval has been run at all this case (a boolean over state — no string
    matching). Used only to enforce that the case is grounded before a first answer and to
    fire the structural high-risk trigger once. Deciding whether *more* evidence is needed
    for a new or pivoted diagnosis is left to the LLM (it sees the evidence and chooses
    action=retrieve), not to any term/keyword/query comparison."""
    return bool(form.get("retrieved"))


def auto_retrieval_target(form):
    """Built from the current chief complaint + the live (high/medium) diagnosis names.

    The diagnosis names are clean clinical terms and match how evidence articles are titled,
    so the query stays anchored on the conditions actually being pursued. A free-text
    narrative is deliberately NOT used: it carries keywords from rejected diagnoses and the
    patient's own guess, which pollute a lexical search."""
    cc = form.get("cc", "")
    dx_names = "; ".join(h.get("dx", "") for h in live_hypotheses(form) if h.get("dx"))
    systems = " ".join(form.get("frame", {}).get("systems", []))
    parts = [cc, dx_names, systems, "approach evaluation differential diagnosis"]
    return " ".join(p for p in parts if p).strip()


def apply_verification(hypotheses, verified):
    """Drop refuted candidates. The verifier is given the exact candidate names and echoes
    them back, so matching is by exact identity (case-insensitive) — never fuzzy/term
    overlap. A candidate is pruned only on an exact 'refuted' match; anything unmatched is
    kept (safe default)."""
    if not verified:
        return hypotheses
    refuted = {
        (v.get("dx", "") or "").strip().lower()
        for v in verified
        if v.get("verdict") == "refuted"
    }
    kept = [h for h in hypotheses if (h.get("dx", "") or "").strip().lower() not in refuted]
    return kept or hypotheses


# ── formatting helpers — Python builds context, LLM only synthesizes ─────────

def format_conversation(conversation):
    lines = []
    for item in conversation[-20:]:
        role = "Patient" if item.get("role") == "patient" else "Doctor"
        lines.append(f"{role}: {item.get('message', '')}")
    return "\n".join(lines) if lines else "(none)"


def format_evidence(retrieved):
    # Most recent retrievals reflect the current hypotheses after any pivot, so ground
    # on those rather than the earliest (possibly superseded) searches. Items are already
    # rank-fused by the retriever; the LLM judges their relevance during synthesis.
    parts = []
    for item in retrieved[-3:]:
        for ev in item.get("evidence", [])[:2]:
            title = ev.get("title", "")
            text = ev.get("text", "")[:300]
            if title or text:
                parts.append(f"[{title}]\n{text}")
    return "\n\n".join(parts) if parts else "(none retrieved yet)"


def format_facts(facts):
    if not facts:
        return "(none structured yet)"
    return "\n".join(f"- {k}: {v}" for k, v in facts.items())


def format_hypotheses(hypotheses):
    if not hypotheses:
        return "(none yet)"
    lines = []
    for h in hypotheses:
        dx = h.get("dx", "")
        lk = h.get("likelihood", "")
        ac = h.get("acuity", "")
        tag = f"{lk}, {ac}" if ac else lk
        lines.append(f"- {dx} ({tag})")
        anchors = ", ".join(h.get("anchors", []))
        missing = ", ".join(h.get("missing", []))
        if anchors:
            lines.append(f"  supports: {anchors}")
        if missing:
            lines.append(f"  missing: {missing}")
    return "\n".join(lines)


def format_candidates(hypotheses):
    return "\n".join(f"- {h.get('dx', '')} (current likelihood {h.get('likelihood', '')})" for h in hypotheses)
