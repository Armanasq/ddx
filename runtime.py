from ddx.config import Config
from ddx.intake import (
    apply_verification,
    auto_retrieval_target,
    build_frame,
    decomposed_queries,
    generate_ddx_answer,
    has_attempted_retrieval,
    is_evidence_relevant,
    live_hypotheses,
    needs_strict_verification,
    structural_retrieval_trigger,
    update_hypotheses,
    verify_candidates,
)
from ddx.cost import CostLedger
from ddx.llm import LLM
from ddx.retrieval import Retriever
from ddx.store import Store, merge_frame, merge_update


class Runtime:
    def __init__(self, session_id="session", trace_llm=False, init_form=False):
        self.config = Config()
        self.store = Store(self.config.storage_dir, session_id)
        if init_form:
            self.store.init()
        self.ledger = CostLedger(self.config.storage_dir)
        self.llm = LLM(self.config, trace=trace_llm, ledger=self.ledger, session_id=session_id)
        self.retriever = Retriever(self.config)

    def turn(self, user_message):
        self.llm.reset()
        form = self.store.load()

        # ── 1. LLM proposes: extract/update the clinical frame ───────────────
        if not form.get("cc"):
            result = build_frame(self.llm, user_message)
            form = merge_frame(form, result)
        else:
            result = update_hypotheses(self.llm, form, user_message)
            form = merge_update(form, result)

        proposed = result.get("action", "ask")
        question = result.get("q", "")
        need = result.get("need", "unknown")
        intent = result.get("user_intent", "report")
        reply = (result.get("reply_to_user") or "").strip()

        # The patient explicitly asked for the assessment → produce the DDx now.
        if intent == "request_assessment" and form.get("cc"):
            proposed = "answer"

        # ── 1b. Pure clarification question: answer it, re-pose the pending
        #         discriminator, and do NOT advance the workup or fire a DDx. ─
        if intent == "clarification" and form.get("cc"):
            pending = _pending_question(form) or question
            shown = f"{reply}\n\n{pending}".strip() if reply else pending
            if pending:
                form["asked"].append({"need": need, "q": pending})
            form["conversation"].append({"role": "patient", "message": user_message})
            if shown:
                form["conversation"].append({"role": "system", "message": shown})
            self.store.save(form)
            return self._result(shown, form)

        # ── 2. Python controls: resolve the real action against the frame ────
        action = self._resolve_action(form, proposed, question)

        # ── 3. Execute ───────────────────────────────────────────────────────
        if action == "answer":
            question = self._produce_ddx(form)
            form["ddx"] = question
        else:  # ask (possibly preceded by a forced retrieval)
            if not question:
                question, need = _fallback_question(form)
            form["asked"].append({"need": need, "q": question})
            # Mixed intent: the patient also asked something — answer it first.
            if intent == "mixed" and reply:
                question = f"{reply}\n\n{question}"

        # ── 4. Persist conversation (ground truth for next turn) ─────────────
        form["conversation"].append({"role": "patient", "message": user_message})
        if question:
            form["conversation"].append({"role": "system", "message": question})

        self.store.save(form)
        return self._result(question, form)

    def _result(self, question, form):
        ledger = self.ledger.totals()
        return {
            "question": question,
            "form": form,
            "llm_calls": self.llm.calls,
            "llm_totals": self.llm.totals(),
            "cost_session": ledger["sessions"].get(self.store.path.stem, {}),
            "cost_all_time": ledger["all_time"],
        }

    # ── controller ───────────────────────────────────────────────────────────

    def _resolve_action(self, form, proposed, question):
        """Map the LLM's proposed action to the real action.

        Decisions come only from the LLM's own action and from enum/boolean state — never
        from term/keyword/query string matching:
          - the case must be grounded (retrieval attempted) before the first answer;
          - a complex/high-risk frame forces the first retrieval (enum-based trigger);
          - the LLM owns everything else: it requests retrieval when it judges more
            evidence is needed (including after a pivot), and it returns answer when it
            cannot find a genuinely new question (it sees the full conversation).
        """
        # No chief complaint yet → ask for it, never retrieve/answer on noise.
        if not form.get("cc"):
            return "ask"

        # Answer: grounding for the final synthesis is guaranteed in _produce_ddx, which
        # always retrieves for the current hypotheses (so a pivoted differential is grounded
        # without any pivot-detection string comparison).
        if proposed == "answer":
            return "answer"

        # Structural high-risk trigger fires the first grounding retrieval (enum-only read).
        if structural_retrieval_trigger(form) and not has_attempted_retrieval(form):
            self._retrieve(form)
            return "ask"

        # The LLM explicitly requested evidence (its judgement, incl. after a pivot).
        if proposed == "retrieve":
            self._retrieve(form)
            return "ask"

        return "ask"

    def _retrieve(self, form):
        target = auto_retrieval_target(form)
        if not target:
            return
        evidence = self.retriever.search(target)
        # Low-cost relevance gate: keep only passages a cheap true/false call confirms
        # are relevant to this patient and the decomposed queries. Drops recall noise
        # (off-topic or wrong-population passages) without any keyword rules.
        queries = decomposed_queries(form)
        context = form.get("frame", {}).get("one_liner", "") or form.get("cc", "")
        relevant = [e for e in evidence if is_evidence_relevant(self.llm, context, queries, e)]
        form["retrieved"].append(_pack_retrieved(target, relevant, form))

    def _produce_ddx(self, form):
        # Invariant: always ground the final answer on evidence for the CURRENT hypotheses.
        # The target is rebuilt from the current diagnosis names, so a pivoted differential
        # is freshly grounded — no pivot detection or coverage string-matching required.
        self._retrieve(form)
        # Strict lane: verify each candidate against evidence and prune the refuted.
        if needs_strict_verification(form):
            verified = verify_candidates(self.llm, form)
            form["hypotheses"] = apply_verification(form.get("hypotheses", []), verified)
        ddx_text = generate_ddx_answer(self.llm, form)
        source_block = _source_block(form)
        return f"{ddx_text}\n\n{source_block}" if source_block else ddx_text


# ── helpers ──────────────────────────────────────────────────────────────────

def _pending_question(form):
    """The last discriminator we asked the patient (still unanswered when they ask us
    something). Pure state lookup — no string matching."""
    for item in reversed(form.get("asked", [])):
        if isinstance(item, dict) and item.get("q"):
            return item["q"]
    return ""


def _fallback_question(form):
    asked_needs = {item.get("need", "") for item in form.get("asked", []) if isinstance(item, dict)}
    facts = form.get("facts", {})
    for hyp in live_hypotheses(form):
        for key in hyp.get("missing", []):
            if key and key not in facts and key not in asked_needs:
                return f"Can you tell me about {key.replace('_', ' ')}?", key
    return "Can you describe any other symptoms you are experiencing?", "other_symptoms"


def _pack_retrieved(target, evidence, form):
    return {
        "target": target,
        "chunks": [item["chunk_id"] for item in evidence],
        "evidence": [
            {
                "id": item["chunk_id"],
                "title": item["title"],
                "path": item["section_path"],
                "text": item["text"][:500],
                "fit": item.get("fit", 0),
            }
            for item in evidence[:3]
        ],
    }


def _source_block(form):
    seen = set()
    titles = []
    for item in form.get("retrieved", []):
        for ev in item.get("evidence", []):
            title = ev.get("title", "").strip()
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
    if not titles:
        return ""
    return "\n".join(["---", "**Sources**"] + [f"- {t}" for t in titles])
