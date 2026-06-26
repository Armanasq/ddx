import json


def frame_prompt(user_message):
    return f"""
You are a clinical reasoning engine. A patient has sent a message. Construct the full clinical frame.

Patient message:
{user_message}

Return JSON only:
{{
  "cc": "",
  "facts": {{}},
  "frame": {{
    "one_liner": "",
    "systems": [],
    "time_course": "acute|subacute|chronic|episodic|unknown",
    "severity": "high|medium|low|unknown"
  }},
  "hypotheses": [
    {{
      "dx": "",
      "likelihood": "high|medium|low",
      "acuity": "emergent|urgent|routine",
      "anchors": [],
      "missing": []
    }}
  ],
  "action": "ask",
  "need": "",
  "q": ""
}}

Rules:
- cc: concise normalized chief complaint. If multiple presenting symptoms, capture all in one phrase.
- facts: every clinical fact stated. Keys: short snake_case. Values: concise strings, never arrays. Denials → "no". Precipitating events → "trigger". Time expressions → "onset".
- frame.one_liner: one clinical synthesis sentence (the problem representation).
- frame.systems: every organ system plausibly involved, listed separately (e.g. ["cardiovascular", "respiratory"]).
- frame.time_course and frame.severity: your best clinical read; use "unknown" only when truly indeterminate.
- hypotheses: 3 to 6 SPECIFIC named diagnoses ranked by likelihood — never vague category labels like "secondary cause" or "structural process". Name the actual conditions.
  - acuity: emergent = immediately life-threatening if untrue and missed; urgent = needs same-day evaluation; routine = can be worked up electively. Classify honestly per diagnosis.
  - anchors: facts supporting it. missing: 2 to 4 snake_case discriminators not yet known that would most change its ranking.
- action: answer if already rich enough for a defensible differential; retrieve if evidence is needed to choose the next move; ask otherwise. (Python may override this.)
- need: snake_case key of the most important unknown.
- q: one focused patient-facing question separating the top hypotheses. Always fill q.
""".strip()


def hypothesis_update_prompt(cc, conversation, facts_text, hypotheses_json, evidence_text, user_message):
    return f"""
You are a clinical reasoning engine running a doctor-led differential workup.

Chief complaint: {cc or "(not yet established)"}

Conversation so far (complete history):
{conversation}

Clinical facts already recorded (key: value):
{facts_text}

New patient message: {user_message}

Retrieved clinical evidence:
{evidence_text}

Current hypotheses:
{hypotheses_json}

Return JSON only:
{{
  "user_intent": "report|clarification|request_assessment|mixed",
  "reply_to_user": "",
  "problem_representation": "",
  "new_facts": {{}},
  "hypotheses": [],
  "action": "ask",
  "need": "",
  "q": ""
}}

Rules:
- user_intent: classify the NEW patient message.
  - report: it only gives clinical information (symptoms, history, answers to your question).
  - clarification: it asks a side question (e.g. "what could this be?", "should I worry?", "what tests?") without asking you to conclude, and gives no new clinical fact.
  - request_assessment: it explicitly asks for your assessment, diagnosis, opinion, or what is most likely / wants you to conclude now.
  - mixed: it both gives new clinical information AND asks a side question.
- reply_to_user: if intent is clarification or mixed, a brief, honest, doctor-like answer to what they asked, grounded in the current hypotheses/evidence (1-3 sentences). Empty string otherwise.
- problem_representation: one current synthesis sentence reflecting EVERYTHING known. Rewrite it each turn; never echo an earlier vaguer version.
- new_facts: clinical facts from the new message ONLY. Empty if intent is clarification or request_assessment. Keys: short snake_case. Values: concise strings, never arrays. Denials → "no". Never store the patient's questions, worries, or meta-requests as facts.
  - CORRECTION HANDLING: if the message corrects an earlier value, reuse the EXACT existing key shown in the recorded facts above so the new value overwrites the old one — do not create a near-duplicate key for the same clinical dimension.
- hypotheses: the complete updated list of SPECIFIC named diagnoses (dx, likelihood high|medium|low, acuity emergent|urgent|routine, anchors, missing). Update likelihoods from the full conversation; drop only when genuinely excluded; add new ones the facts/evidence raise. In missing, list only discriminators not yet discussed.
- action:
  - answer: the differential is clear and well-separated and no single new question would dramatically change the top of it. (Evidence grounding is added automatically before the final answer — do not keep asking just to gather evidence.)
  - retrieve: a newly raised or pivoted high/medium diagnosis would benefit from evidence to guide the next question. Prefer this over asking when the syndrome has shifted.
  - ask: one new question would meaningfully change the ranking.
  - If intent is question (no new info), use ask — you are not advancing the workup, just answering then continuing.
- need: short snake_case label for the discriminator q targets.
- q: one focused question on a topic provably absent from the conversation above; never repeat or rephrase a question already asked. If intent is question, q is the discriminator you still need next. If you cannot find a genuinely new useful question, return action=answer.
""".strip()


def verification_prompt(problem_representation, candidates_text, evidence_text):
    return f"""
You verify candidate diagnoses against retrieved clinical evidence before a final differential is issued.

Problem representation:
{problem_representation}

Candidate diagnoses:
{candidates_text}

Retrieved clinical evidence:
{evidence_text}

For each candidate, judge ONLY against the evidence and problem representation above — not your unaided memory.

Return JSON only:
{{
  "verified": [
    {{ "dx": "", "verdict": "supported|insufficient|refuted", "rationale": "" }}
  ]
}}

Rules:
- supported: the evidence is consistent with this diagnosis given the presentation.
- insufficient: the evidence neither supports nor excludes it; it stays on the differential as uncertain.
- refuted: the evidence or presentation clearly argues against it; it should be dropped.
- rationale: one short clause citing the deciding feature.
- Judge every candidate. Do not invent diagnoses not listed.
""".strip()


def relevance_prompt(patient_context, queries_text, title, text):
    return f"""A clinical reasoning system retrieved a passage. Decide if it is relevant to THIS patient.

Patient: {patient_context}

Clinical question and candidate diagnoses being worked up:
{queries_text}

Retrieved passage:
Title: {title}
{text}

Is this passage relevant and useful for reasoning about this patient's question or any of those
diagnoses (consider the patient's context, e.g. age group, so a passage about a different
population is not relevant)? Reply with one word only: true or false.""".strip()


def ddx_answer_prompt(cc, facts_text, conversation, hypotheses_text, evidence_text):
    return f"""
You produce the final differential diagnosis synthesis.

Chief complaint: {cc}

Clinical facts collected:
{facts_text}

Full conversation:
{conversation}

Verified ranked hypotheses (refuted candidates already removed):
{hypotheses_text}

Retrieved clinical evidence:
{evidence_text}

Write a clinician-facing answer. Not JSON.

Format:
1. Start with: "This is not a diagnosis."
2. Problem representation — one sentence.
3. Key findings — most important supporting facts and relevant negatives.
4. Ranked differential — the specific named diagnoses above. For each: likelihood (high/medium/low), why it fits, what argues against it, what would confirm or exclude it.
5. Next steps — recommended tests/evaluation with rationale, and the single most important next action. State urgency honestly based on the most acute live diagnosis.

Rules:
- No numeric confidence scores.
- Use specific named diagnoses, never vague category labels.
- Do not present any single diagnosis as certain.
- Use only the facts, conversation, and evidence above.
- If evidence is weak or mismatched, say so explicitly and keep the differential uncertain.
""".strip()
