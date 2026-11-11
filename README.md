# Manga Tracker Agent

**CS4680 Prompt Engineering — AI Agent Project**

An AI agent that reads your plain-English messages, turns them into structured
actions with an LLM, validates those actions, and executes them against a
database, two live manga APIs, and the file system.

```
"I'm on ch 142 of Kagurabachi, drop Boruto, and add Dandadan to plan-to-read"
```

becomes three validated, logged, individually-confirmable actions.

---

## The problem it solves

Manga readers follow a dozen or more ongoing series released on irregular
schedules by different scanlation groups. Keeping track means either manual
spreadsheet upkeep or clicking through a tracking site's forms one series at a
time. Neither answers the question people actually have: *what do I have left
to read right now?*

This agent takes a sentence and does the bookkeeping, then checks MangaDex for
chapters released past your progress.

---

## How the agent works

```
Your message
   ↓
Prompt assembly       system prompt + 6 few-shot examples + your live library
   ↓
LLM (Gemini)          returns {reasoning, reply, actions[]} as JSON
   ↓
Validator             schema check; invalid output is fed back to the LLM
   ↓                  with the error so it can repair itself (up to 2 retries)
Safety gate           destructive actions wait for an explicit click
   ↓
Executor              database / manga APIs / file export
   ↓
Audit log             every attempt recorded, success or failure
```

---

## Project requirements, and where each one lives

### 1. LLM integration module — `agent/llm_client.py`

- Google Gemini through the official `google-genai` SDK, behind a provider
  interface so another vendor can be dropped in.
- `response_mime_type="application/json"` and `temperature=0.1`, because the
  output is parsed by a program, not read by a human.
- **Rate limiting and error handling:** `classify_error()` sorts provider
  exceptions into retryable (429, quota, 5xx, timeout) and fatal (bad key).
  `call_with_backoff()` retries retryable failures with exponential backoff
  plus jitter, and does *not* waste retries on an invalid API key.
- `MockLLM` lets the whole app and test suite run with no key and no network.

### 2. Action interpreter / executor — `agent/schemas.py`, `agent/validator.py`, `agent/executor.py`

Eleven actions across three domains:

| Domain | Actions |
|---|---|
| **Database** | `add_series`, `update_progress`, `set_status`, `rate_series`, `remove_series`, `query_library` |
| **Manga APIs** | `search_manga`, `check_new_chapters`, `get_recommendations` |
| **Files** | `export_list` (CSV/Markdown/JSON), `import_list` (CSV) |

- `schemas.py` defines each action as a Pydantic model with `extra="forbid"`,
  so an unexpected key is an error rather than something silently ignored.
- `validator.py` extracts the JSON (handling markdown fences and surrounding
  prose), validates it, and caps the plan length.
- `executor.py` dispatches each action. One action failing does not abort the
  rest of the plan.

### 3. User interface — `app.py` (Streamlit)

- Accounts: sign up / sign in, with each account's library kept private.
- Chat input for natural language, with example prompts for first-time users.
- Every turn shows the **plan**: each action, its status icon, and what it did.
- Live sidebar: library grouped by status, a "new chapters waiting" section,
  the action log, CSV import, and log download.
- Metrics row: series tracked, currently reading, new chapters, chapters read.
- Confirmation cards for destructive actions.

### 4. Error handling and safety — `agent/validator.py`, `agent/logger.py`

- **Output validation.** Unknown action names, missing fields, wrong types,
  negative or absurd chapter numbers, out-of-range ratings, invalid statuses,
  and unexpected extra keys are all rejected before execution.
- **Confirmation for destructive operations.** `remove_series`, dropping a
  series, and `import_list` require a click. Critically, the
  `requires_confirmation` flag is **re-derived by the validator**, not taken
  from the LLM — a model claiming `"requires_confirmation": false` on a delete
  cannot bypass the prompt. (`DESTRUCTIVE` is a `ClassVar`, so it is not even a
  field the model can set.) On confirmation the **stored plan** is executed
  directly without calling the LLM again, so the approved actions cannot change
  between the preview and the click.
- **Graceful failure.** API outages, rate limits, timeouts, malformed JSON,
  missing series and bad CSV rows all produce a readable message. Any unhandled
  exception inside an action is caught, rolled back, and reported.
- **Action cap.** At most 8 actions per request.
- **External API politeness.** Minimum intervals keep requests under
  MangaDex's ~5/sec and AniList's current 30/min limits, with `Retry-After`
  honoured on 429 and a 15-minute response cache.
- **Audit log.** Every action is stored with timestamp, the original request,
  the action JSON, the outcome, and whether the user confirmed it. Viewable in
  the sidebar and downloadable as CSV.

---

## Prompt engineering techniques

All in `agent/prompts.py`, which documents each one inline.

| Technique | What it does | Why it was needed |
|---|---|---|
| **Role + constrained tool vocabulary** | The system prompt enumerates all 11 actions with argument schemas | Without it the model invents plausible-sounding action names |
| **Structured output contract** | Requires one JSON object: `reasoning`, `reply`, `actions` | Prose cannot be executed |
| **Plan-before-act** | The `reasoning` field is written before the actions | Improves multi-step requests; capped so it cannot ramble |
| **Few-shot examples** | Six worked examples: multi-action, nickname resolution, destructive, filtered recommendation, and a "don't guess, ask" case | Teaches output shape far better than description |
| **Context injection** | The user's live library is rendered into every prompt | Lets "caught up on OP" resolve to `One Piece` at its real latest chapter instead of a hallucinated number |
| **Negative constraints** | Explicit rules: never invent a title for an update, ask when ambiguous, never claim an action already happened | Removes the most common failure modes |
| **Self-correction loop** | Validation errors are fed back with the model's own bad output and a repair instruction | Turns a hard failure into a recovered turn — covered by tests |
| **Low temperature + JSON mime type** | Deterministic, parseable output | The response is consumed by a parser |

The self-correction loop is the piece worth demoing: ask for something the
model gets wrong on the first try and the app recovers without the user seeing
an error.

---

## Running it locally

```bash
git clone <your-repo-url>
cd manga_agent
pip install -r requirements.txt
streamlit run app.py
```

With no configuration it uses a local SQLite file and runs the LLM in demo
mode. To enable the agent, create `.streamlit/secrets.toml`
(see `.streamlit/secrets.toml.example`):

```toml
GEMINI_API_KEY = "your-key-from-aistudio.google.com"
```

Then verify everything before launching:

```bash
python selfcheck.py
```

It checks the database, the LLM key, MangaDex and AniList independently, so a
failure points at exactly one thing.

---

## Deploying

### 1. Gemini API key
Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey), create
a key, and copy it. The free tier is enough for this project.

### 2. Neon Postgres database
Streamlit Community Cloud's filesystem is **ephemeral** — a SQLite file there
is wiped on every restart and redeploy, which would erase every user's library.
So the database lives outside the app.

1. Sign up at [neon.com](https://neon.com) (free, no card).
2. Create a project; copy the connection string from the dashboard.
3. It looks like
   `postgresql://user:password@ep-xxx.region.aws.neon.tech/neondb?sslmode=require`

Neon's free tier keeps the database reachable while idle, so the app does not
go dark between demos.

### 3. GitHub
```bash
git init
git add .
git commit -m "Manga tracker AI agent"
git branch -M main
git remote add origin https://github.com/<you>/manga-tracker-agent.git
git push -u origin main
```

`.gitignore` already excludes `.streamlit/secrets.toml` and `*.db`, so no
credentials are committed. **Confirm with `git status` before your first push.**

### 4. Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
2. "Create app" → pick your repo → main file `app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   DATABASE_URL = "your-neon-connection-string"
   ```
4. Deploy. Every push to `main` redeploys automatically.

---

## Tests

```bash
python -m pytest -q
```

202 tests, no network or API key required.

Note: Google retires Gemini model IDs on a published schedule. The default is
`gemini-3.6-flash` (stable as of November 2026). If it is ever retired, the app
says so explicitly and `selfcheck.py` prints the models your key can reach, so
the fix is setting `LLM_MODEL` in your secrets.

| File | Covers |
|---|---|
| `test_auth.py` | Password hashing and salting, login, and that one account cannot read or modify another's library |
| `test_validator.py` | JSON extraction from fenced/prose-wrapped output, unknown actions, every argument constraint, and that the destructive flag cannot be overridden by the model |
| `test_llm_client.py` | Error classification, exponential backoff, no retries on a bad key |
| `test_manga_api.py` | MangaDex and AniList parsing, throttling, caching, 429/5xx/timeout handling, driven by canned HTTP responses |
| `test_executor.py` | All 11 actions, fuzzy title matching and its ambiguity guard, CSV sanitising, partial plan failure |
| `test_agent.py` | The full loop: multi-action turns, the confirmation gate, the self-correction retry, the action cap, prompt assembly |
| `test_ui.py` | The real `app.py` driven through Streamlit's `AppTest`: signup, login, chat turns, tables, downloads, confirm/cancel, and account separation |
| `test_config_db.py` | Secret resolution and Neon/Postgres connection-string normalisation |

---

## Demo video outline (1–2 min)

1. **Sign up** — show that the account is yours and starts empty.
2. **One sentence, three actions** — paste the Kagurabachi/Boruto/Dandadan
   example. Point at the plan panel: three actions, two applied, one waiting.
3. **Confirm the destructive one** — show the warning card, click Confirm,
   watch the sidebar update.
4. **Release watching** — "anything new to read?" Live MangaDex data showing
   how far behind you are.
5. **Recommendation** — "something like Vinland Saga but shorter and finished."
6. **Safety** — open the action log, download it. Mention that a delete always
   asks, and that the LLM cannot opt out of that.

`sample_import.csv` is included if you want to demo CSV import.

---

## Layout

```
app.py                  Streamlit UI
selfcheck.py            Pre-flight credential/service check
requirements.txt
sample_import.csv       Demo CSV for import_list
agent/
  config.py             Settings from env or Streamlit secrets
  db.py                 SQLAlchemy models (users, series, action_log)
  auth.py               Signup/login, bcrypt hashing
  llm_client.py         Gemini wrapper, retry/backoff, MockLLM
  prompts.py            System prompt, few-shot examples, context injection
  schemas.py            Pydantic schema for all 11 actions
  validator.py          JSON extraction, validation, safety flags
  manga_api.py          MangaDex + AniList, throttled and cached
  executor.py           Action dispatch across the three domains
  logger.py             Audit log
  agent.py              The orchestration loop
tests/                  202 tests
```

## Credits

Chapter data from [MangaDex](https://mangadex.org); recommendation data from
[AniList](https://anilist.co). Both are community-run services used within
their published rate limits.
