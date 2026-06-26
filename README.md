# ddx

A doctor-like differential-diagnosis (DDx) engine. It reasons over a multi-turn
conversation, retrieves clinical evidence when the case warrants it, verifies its
candidates against that evidence, and produces a ranked, evidence-grounded differential.

The design separates two responsibilities:

- **The LLM** does clinical interpretation: building the case frame, generating and
  ranking specific named diagnoses, classifying their acuity, extracting facts, and
  phrasing questions and answers.
- **Python** owns control flow and state: when to retrieve, when enough evidence exists
  to answer, hypothesis-coverage bookkeeping, loop prevention, and persistence.

There is **no case-specific logic** anywhere in Python: no disease `if/else`, no
symptom→diagnosis tables, no keyword or pattern matching in the reasoning path. Behaviour
emerges from the LLM-produced case frame plus structural policy over its enum fields.

## Run

```bash
python -m ddx.app --session demo            # interactive
python -m ddx.app --session demo --llm-call # also print per-call token/cost stats
python -m ddx.app --session demo --debug    # also print the full form each turn
```

Type `exit`, `quit`, or `q` to stop. Each session writes a full transcript (LLM I/O,
token stats, form state per turn) to `ddx/storage/transcripts/<timestamp>_<session>.md`.

## Per-turn pipeline (`Runtime.turn`)

```text
user message
  -> load form from ddx/storage/<session>.json
  -> LLM step:
       first turn  : build_frame()        -> cc, facts, frame, hypotheses, action, q
       later turns : update_hypotheses()  -> user_intent, reply, new_facts,
                                             updated hypotheses, action, q
  -> merge result into the form
  -> if the message was a pure clarification question:
       answer it, re-pose the pending question, do NOT advance the workup
  -> else: Python controller resolves the real action (ask | retrieve | answer)
  -> execute it (retrieve / produce DDx / ask)
  -> append the exchange to the conversation history
  -> save the form
```

## The form (`store.py`)

```json
{
  "wf": "ddx",
  "session_id": "",
  "cc": "",
  "frame": { "one_liner": "", "systems": [], "time_course": "", "severity": "" },
  "hypotheses": [
    { "dx": "", "likelihood": "high|medium|low",
      "acuity": "emergent|urgent|routine", "anchors": [], "missing": [] }
  ],
  "facts": {},
  "conversation": [ { "role": "patient|system", "message": "" } ],
  "asked": [ { "need": "", "q": "" } ],
  "retrieved": [
    { "target": "", "chunks": [],
      "evidence": [ { "id": "", "title": "", "path": "", "text": "", "fit": 0 } ] }
  ],
  "ddx": ""
}
```

`conversation` is the ground truth the LLM reasons from each turn — not the `facts` dict.
`facts` is a structured record (used for the final synthesis and for correction/overwrite
of values); the multi-turn reasoning works from the full conversation so that imperfect
extraction cannot derail it.

## LLM calls (`prompts.py`, `intake.py`)

| Call | When | Returns |
|------|------|---------|
| `frame_prompt` | first turn | cc, facts, `frame`, ranked `hypotheses` (with `acuity`), proposed `action`, `need`, `q` |
| `hypothesis_update_prompt` | later turns | `user_intent`, `reply_to_user`, `problem_representation`, `new_facts`, updated `hypotheses`, `action`, `need`, `q` |
| `verification_prompt` | before a DDx in the strict lane | per-candidate verdict: `supported` / `insufficient` / `refuted` |
| `ddx_answer_prompt` | final synthesis | clinician-facing ranked differential text |

Every prompt is built from structured fields via Python f-strings; the LLM is never asked
to regenerate something Python already holds (e.g. the retrieval query is built in Python,
not by the LLM).

## Python control policy (`intake.py`, `runtime.py`)

All of the following read only LLM-produced enum/structured fields — never message text:

No control decision uses term/keyword/query string matching. Decisions come from the LLM's
own `action`, from enum reads of the frame, and from booleans over state:

- **`structural_retrieval_trigger`** — fires the first grounding retrieval when the frame is
  complex or high-risk: ≥2 organ systems, or `time_course == "acute"` and
  `severity == "high"`, or a high/medium hypothesis whose `acuity` is `emergent`/`urgent`.
  This reads only enum fields.
- **`has_attempted_retrieval`** — a boolean over state (has any retrieval run yet). It gates
  the structural trigger so it fires once; it does no name/coverage matching.
- **`auto_retrieval_target`** — builds the search query from the chief complaint + the
  current high/medium diagnosis names + involved systems. (Constructing the query string is
  not a decision; it is how the index is searched.)
- **`needs_strict_verification`** — complex/high-risk cases get candidate-by-candidate
  verification (`verify_candidates` → `apply_verification` drops `refuted` ones by exact
  diagnosis identity); simple cases answer directly.

Controller (`Runtime._resolve_action`), all decisions LLM- or enum/boolean-driven:

- the LLM's `action` (`ask`/`retrieve`/`answer`) is honoured; the LLM requests retrieval
  when it judges more evidence is needed, including after a pivot;
- the structural trigger forces the first grounding retrieval for high-risk frames;
- **grounding is guaranteed at answer time** — `_produce_ddx` always retrieves for the
  *current* hypotheses before synthesising, so a pivoted differential is grounded without
  any pivot-detection or coverage string comparison;
- whether a question repeats is left to the LLM (it sees the full conversation and is told
  not to repeat); there is no token-overlap loop-breaker.

Clarification handling: `user_intent` is one of `report` / `clarification` /
`request_assessment` / `mixed`. On `clarification` the engine answers the user
(`reply_to_user`), re-poses the pending discriminator, and does not advance the workup or
store the question as a fact. On `request_assessment` it produces the DDx now. `mixed`
messages are answered and also processed as clinical
information.

## Retrieval (`retrieval.py`)

Hybrid search with **no keyword/term/pattern logic**:

- `sql_search` — SQLite FTS5 BM25 (lexical).
- `qdrant_search` — dense-vector semantic search (embeddings via the configured endpoint).
- `reciprocal_rank_fusion` — fuses the two ranked lists using only rank positions
  (`score = Σ 1/(k + rank)`, `k = 60`). Items both retrievers rank highly rise; single-
  retriever recall noise sinks. The fused score is stored per chunk as `fit`.

There is no hand-coded reranker. After fusion, each candidate passage goes through a
**low-cost relevance gate** (`is_evidence_relevant` → `relevance_prompt`): a cheap call
that is given the patient context and the decomposed queries (chief complaint + each live
diagnosis name) and replies with a single word, `true` or `false`. Only passages confirmed
`true` are kept as evidence. This drops off-topic and wrong-population recall noise without
any keyword rules — the judgement is the model's, not a hand-coded filter.

## Cost tracking (`cost.py`)

Every LLM call is priced and recorded. `llm.py` computes a per-call breakdown
(`input_cost_usd`, `cached_cost_usd`, `output_cost_usd`, `cost_usd`) from the prices in
`config.py`, and `CostLedger` folds each call into two files under the storage dir:

- `cost_ledger.json` — rolling cumulative totals: `all_time` (the whole development cost
  across every session and test run) and a per-`session` breakdown, each with call count,
  token counts, and the input/output cost split. Written atomically (temp file +
  `os.replace`).
- `cost_log.jsonl` — append-only audit log, one line per LLM call.

Because the ledger is wired into `Runtime` (and therefore every LLM call, from the app or
from test scripts), the `all_time` total is a running sum of all usage over time. Each
turn's result also carries `cost_session` and `cost_all_time`, and the CLI prints a
`[COST] turn=… session=… all_time=…` line.

## Configuration (`config.py`, `.env`)

Read from `ddx/.env` (or process env):

- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`,
  `DDX_AZURE_DEPLOYMENT` — chat model.
- `EMBEDDING_URI`, `EMBEDDING_AUTH` — embedding model for dense retrieval.
- `QDRANT_URL`, `QDRANT_CHUNKS_COLLECTION` — vector store.
- `DDX_KNOWLEDGE_DB` — SQLite knowledge base (chunks/documents/sections + `chunks_fts`).
- `DDX_STORAGE_DIR` (default `ddx/storage`), `DDX_MAX_LLM_TOKENS`.

## Module map

| File | Responsibility |
|------|----------------|
| `app.py` | CLI loop |
| `runtime.py` | per-turn orchestration + control policy |
| `intake.py` | LLM-call wrappers, structural policy, context formatting |
| `prompts.py` | the four prompt templates |
| `retrieval.py` | hybrid BM25 + dense retrieval, RRF fusion |
| `store.py` | form schema, load/save, merge of frame and update results |
| `llm.py` | Azure OpenAI client, JSON/text completion, per-call token-cost accounting |
| `cost.py` | persistent cost ledger (all-time + per-session) and per-call audit log |
| `transcript.py` | per-session markdown transcript |
| `config.py` | environment/config loading |
