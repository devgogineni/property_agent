# Estate Agent — UK Property RAG Chatbot

A retrieval-augmented chatbot that answers questions about UK properties
("what have houses on this street sold for?", "how are prices trending in
this area?") by grounding OpenAI LLM answers in two real government
datasets: **Price Paid Data** (individual historical property sales) and the
**UK House Price Index** (regional price trends).

## Architecture

```
data ingestion  ─▶  durable store (SQLite)  ─▶  RAG agent  ─▶  FastAPI (JSON + HTML)
```

1. **Ingestion** (`ingestion/`) loads both raw CSVs into a local SQLite
   database (`data/estate_agent.db`) and builds a small location-name
   resolver index.
2. **Durable store**: retrieval here is fundamentally structured (postcode /
   street / locality / town / district / county lookups over up to 31M
   rows), not open-ended semantic search — so **SQLite** is the durable
   store, not a vector DB. It's file-based, indexable, and handles that
   volume comfortably.
3. **Location resolver**: `minsearch.VectorSearch` + the ONNX MiniLM
   embedder (`utils/embedder.py`) power a small fuzzy-matching index built
   only over the *distinct* locality/town/district/county/region names
   (thousands, not millions) — used to map a user's free text ("Newcastle")
   to the canonical value stored in the data before running an exact SQL
   filter. Postcode/street are excluded (too high-cardinality, not
   meaningfully fuzzy) and matched directly via SQL instead.
4. **RAG agent** (`rag/agent.py`): extracts structured filters from the
   user's question via an OpenAI structured-output call, resolves location
   text to canonical values, runs parametrized SQL against both tables,
   builds a context block from the results, and asks the OpenAI LLM to
   answer using only that context.
5. **API** (`api/`): a FastAPI app exposing `POST /chat` (JSON) plus minimal
   built-in HTML pages: a chat UI at `/`, a prompt history page at
   `/prompts`, and a metrics dashboard at `/dashboard`.
6. **Monitoring** (`db/prompt_log.py`, `db/llm_call_metrics.py`,
   `db/dashboard_metrics.py`, `db/feedback.py`): every `/chat` request, each
   individual LLM call within it (entity extraction + final answer + LLM
   judge), any user thumbs up/down feedback, and an automatic LLM-as-judge
   relevance verdict on the answer are all logged to a separate SQLite file,
   powering the prompt history and dashboard pages. See
   [Monitoring & cost tracking](#monitoring--cost-tracking).

## Datasets

| Dataset | File(s) | Notes |
|---|---|---|
| Price Paid Data (PPD) | `data/ppd/pp-YYYY.csv` (one file per year) | Every individual property sale in England & Wales for that year. `ingest_ppd.py` discovers and streams all `pp-YYYY.csv` files in `data/ppd/`, oldest year first; any other files in that folder (e.g. renamed backups) are ignored. |
| UK House Price Index (HPI) | `data/house_price_index_data/UK-HPI-full-file-2026-04.csv` | Monthly average price/index/trend per region, back to the 1960s–70s depending on region (150,705 rows). |

## Project layout

```
config.py                  # env-driven configuration
db/                        # SQLite schema + connection helpers, prompt/metrics logging
utils/                     # embedder, index builder, RAG base class, LLM/cost/latency helpers, data cleaning
ingestion/                 # ingest_hpi.py, ingest_ppd.py, build_location_index.py
rag/                       # schemas, location resolver, SQL retriever, the RAG agent
api/                       # FastAPI app + static chat/prompt-history/dashboard pages
evaluation/                # vector search ground-truth generation + hit_rate/MRR evaluation
scripts/                   # one-off setup scripts (embedding model download)
data/                      # raw CSVs + generated SQLite DBs + location index (gitignored)
models/                    # downloaded ONNX embedding model (gitignored)
```

## Setup

1. **Install dependencies**
   ```
   uv sync
   ```
2. **Add your OpenAI API key** to `.env`:
   ```
   OPENAI_API_KEY=sk-proj-...
   ```
3. **Download the embedding model** (MiniLM ONNX, ~90MB, used for location
   fuzzy-matching):
   ```
   uv run python -m scripts.download_embedding_model
   ```

## Ingestion

```
# HPI: full reload every run, ~150k rows, a few seconds
uv run python -m ingestion.ingest_hpi

# PPD: bounded sample for fast local iteration (default limit: 400,000 rows,
# spread across pp-YYYY.csv files oldest-year-first until the limit is hit)
uv run python -m ingestion.ingest_ppd --limit 400000

# PPD: all year-wise files in data/ppd/ in full - slower and produces a
# multi-GB SQLite file; run this for production use, e.g. as a background job
uv run python -m ingestion.ingest_ppd --limit 0

# PPD: point at a different directory, or ingest a single file directly
uv run python -m ingestion.ingest_ppd --ppd-dir data/ppd --limit 0
uv run python -m ingestion.ingest_ppd --csv data/ppd/pp-2024.csv --limit 0

# Build the location-name resolver index (run after ingesting both datasets)
uv run python -m ingestion.build_location_index
```

`ingest_ppd.py` truncates and reloads the `ppd` table by default; pass
`--append` to add rows without clearing existing ones.

## Running the API

```
uv run uvicorn api.main:app --reload
```

- `GET /health` — row counts for both tables, to confirm ingestion succeeded.
- `GET /` — the browser chat page, with buttons through to the dashboard and
  prompt history pages, +1/-1 feedback buttons under each answer, and a
  colored badge showing the LLM judge's relevance verdict.
- `GET /prompts` — every logged question/answer (with 👍/👎 feedback counts
  and the judge's relevance badge), click a row to expand its per-call
  metrics (extraction + answer + judge: model, latency, tokens, cost).
- `GET /dashboard` — aggregate KPIs (total conversations, avg response time,
  total cost, avg tokens, positive/negative feedback, relevant/partly
  relevant/non-relevant counts) plus cost/response-time-over-time charts for
  the last 100 conversations.
- `POST /chat` — `{"question": "..."}` → grounded answer + retrieval
  metadata, including a `request_id` used to attach feedback to that
  conversation and a `relevance` verdict from the LLM judge:

```
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What have properties sold for recently on Beulah Hill in Croydon?"}'
```

```json
{
  "request_id": 42,
  "answer": "...",
  "extracted_query": {"street": "Beulah Hill", "town": "Croydon", ...},
  "resolved_location": {"ppd": {"field": "town", "value": "CROYDON"}, "region": {...}},
  "num_ppd_matches": 6,
  "num_hpi_matches": 3,
  "total_cost_usd": 0.0021,
  "relevance": "RELEVANT",
  "relevance_explanation": "..."
}
```

- `POST /api/feedback` — `{"request_id": 42, "score": 1}` (or `-1`) records a
  thumbs up/down on that conversation's answer; 404s if `request_id` is
  unknown.

## Running with Docker

The app can run in a container instead of a local `uv` environment. `data/`
(the SQLite DBs, PPD/HPI CSVs, location index) and `models/` (the ONNX
embedding model) are **not** baked into the image — they're bind-mounted
from the host at `./data` and `./models`, since `data/estate_agent.db` alone
can be multi-GB. `config.py`'s paths all resolve relative to the app
directory, so no extra path env vars are needed for this to line up.

1. Complete steps 2–3 of [Setup](#setup) on the host first (`.env` with
   `OPENAI_API_KEY`, and download the embedding model) — the container reads
   `./models` and `./data` from the host, it doesn't fetch them itself.
2. Build and start:
   ```
   docker compose up --build
   ```
3. Run ingestion inside the container (writes to the bind-mounted `./data`,
   so this only needs doing once regardless of whether you run ingestion
   from the host or in Docker):
   ```
   docker compose run --rm api uv run python -m ingestion.ingest_hpi
   docker compose run --rm api uv run python -m ingestion.ingest_ppd --limit 400000
   docker compose run --rm api uv run python -m ingestion.build_location_index
   ```
4. The API is on `http://localhost:8000`, with a container `HEALTHCHECK`
   hitting `GET /health`.

## Monitoring & cost tracking

Every LLM call (entity extraction + final answer) is tracked via
`utils/evaluation_utils.py`'s usage/cost helpers, reused unchanged by the
agent (`RAGWithUsage`). `total_cost_usd` in the `/chat` response reflects the
combined cost of both calls for that request.

Beyond the response itself, each call is timed and persisted for later
inspection, following the [llm-zoomcamp monitoring-metrics
methodology](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/04-metrics.md):
`RAGWithUsage` builds one `LLMCallRecord` (model, prompt, instructions,
answer, token counts, response time, cost) per individual LLM call rather
than a single aggregate per question, so per-call latency isn't lost to
averaging.

This is written to three SQLite tables in a separate database file
(`config.PROMPT_LOG_DB_PATH`, `data/prompt_log.db` by default — kept apart
from `config.DB_PATH` since that one gets deleted/rebuilt by ingestion,
which would otherwise wipe prompt history):

- `prompt_logs` — one row per `/chat` request (question, answer, model,
  aggregate tokens/cost).
- `llm_call_metrics` — one row per individual LLM call (extraction and
  answer), linked back to its `prompt_logs` row via `request_id`, with
  per-call model/prompt/instructions/tokens/response time/cost.
- `feedback` — one row per feedback signal on a `/chat` answer, linked back
  via `request_id`, distinguished by `source`:
  - `source="user"` — thumbs up/down (`score`: `1`/`-1`) submitted from the
    chat page's +1/-1 buttons via `POST /api/feedback`, following the
    [llm-zoomcamp user-feedback
    methodology](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/08-user-feedback.md).
  - `source="judge"` — an automatic `relevance`/`explanation` verdict (see
    below), `score` left `NULL`.

### LLM-as-a-judge relevance check

Following the [llm-zoomcamp built-in-judge
methodology](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/09-built-in-judge.md),
every `/chat` answer is graded inline, immediately after it's generated, by
a second structured-output LLM call (`rag/judge.py`,
`PropertyRAGAgent.judge_relevance()`) that sees only the question and the
generated answer (no reference answer to compare against) and classifies it
as `RELEVANT`, `PARTLY_RELEVANT`, or `NON_RELEVANT` with an explanation.
The verdict is:

- returned in the `/chat` response (`relevance`, `relevance_explanation`)
  and shown as a colored badge under the answer on the chat page;
- stored as a `feedback` row with `source="judge"`;
- logged to `llm_call_metrics` with `call_type="judge"` like any other LLM
  call (model/tokens/latency/cost) — but its cost is deliberately *excluded*
  from `total_cost_usd`/`prompt_logs.cost_usd` (computed before the judge
  runs), so the "cost of serving this answer" KPIs on `/prompts` and
  `/dashboard` aren't inflated by the judging itself; judge cost stays
  fully visible per-call for anyone drilling into a conversation on
  `/prompts`.

The `/prompts` and `/dashboard` pages (and their `/api/prompt-logs`,
`/api/llm-call-metrics`, `/api/dashboard-summary`, and
`/api/dashboard-conversations` JSON endpoints) read from these tables — see
[Running the API](#running-the-api).

## Vector search evaluation

The location-name resolver (see step 3 above) is evaluated with `hit_rate`
and `mrr`, following the [llm-zoomcamp search-evaluation
methodology](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/04-evaluation/lessons)
(`compute_relevance_total` → `hit_rate`/`mrr`):

```
# Generate ground truth: samples distinct location values and uses an LLM to
# synthesize realistic user typos/abbreviations per value (costs a few LLM
# calls), written to data/ground_truth_location.csv
uv run python -m evaluation.generate_ground_truth --sample-size-per-field 20

# Replay those questions through LocationResolver.resolve() and report
# hit_rate@N / MRR@N
uv run python -m evaluation.evaluate_location_search
```

`evaluation/metrics.py` is generic over any `search_function(question,
field) -> list[dict]`, so it isn't tied to this one index if another search
component needs the same evaluation later.

## Known limitations

- A bounded `--limit` fills up from the oldest `pp-YYYY.csv` file forward,
  so a small limit under-represents recent years; run with `--limit 0` for
  comprehensive recent-sales coverage.
- Geographic matching is intentionally permissive: a question mentioning
  any one of town/district/county/locality will match sales where *any* of
  those columns agree, since UK addresses commonly conflate these levels
  (e.g. "Croydon" is colloquially a town but PPD stores it as a `district`,
  with `town=LONDON`).
- HPI (regional trend) lookup is only ever derived from
  town/district/county/locality text, never from a postcode. A
  postcode-only question (e.g. "sales trend for CR0 5NH") returns matching
  PPD sale rows fine, but no HPI trend section, since nothing maps the
  postcode to a region.
- No date-range filtering: `ExtractedQuery` has no date field, so phrases
  like "from 2012 onwards" are silently ignored by both the PPD and HPI
  queries.
