# Localized Agentic Memory Framework

## Project Report

**Prepared:** 27 October 2026  
**Project type:** Final-year project / local-first AI assistant prototype  
**Primary implementation:** Python, Streamlit, Ollama, SQLite, ChromaDB

## 1. Executive summary

The Localized Agentic Memory Framework is a conversational assistant designed
to retain useful information between conversations without relying on a
hosted LLM API. It runs language models through Ollama on the user's computer.
Recent conversation turns are temporarily recorded in SQLite, and a
consolidation process extracts durable user facts for a semantic memory
collection in ChromaDB. When new facts resemble existing ones, a three-band
deduplication process decides whether to skip, add, or replace a stored fact.

The project also provides chat, web search, deep search, memory recall, and
incognito modes. The Streamlit interface is the primary documented interface;
a separate command-line chat program is present as well.

The main technical contribution is the memory pipeline:

```text
Conversation turns
       |
       v
SQLite session buffer
       |
       v
Local LLM fact extraction
       |
       v
Exact-text hash filtering
       |
       v
ChromaDB similarity + three-band decision
       |
       v
Long-term semantic memory
       |
       v
Similarity-based memory retrieval
```

The system is local for language-model inference and local data storage.
However, web-search modes use DuckDuckGo search and, in deep search, fetch
pages from the internet. Therefore, the complete application is not
offline-only: web-search queries and page requests leave the local machine.

## 2. Goals and intended use

The project explores how an AI assistant can maintain useful user context
across sessions while keeping its model inference and memory databases local.
The README positions it as a framework for students, researchers, developers,
and privacy-conscious users interested in local LLMs, conversational memory,
retrieval-augmented generation, and semantic deduplication.

The design goals visible in the implementation are:

- Run inference using locally served Ollama models.
- Keep recent conversation history in a relational session buffer.
- Extract durable, explicit user facts rather than storing every exchange as
  permanent memory.
- Retrieve memories by semantic similarity instead of exact keyword matching.
- Reduce redundant memories and preserve clear corrections or updates.
- Offer distinct interaction modes for chat, web research, memory, and
  non-persistent incognito use.

## 3. Technology stack

| Component | Role in the project |
|---|---|
| Python | Application logic and integration |
| Streamlit | Interactive browser-based chat interface |
| Ollama | Local model serving and inference |
| Qwen 2.5 1.5B | Model selected for the Streamlit chat handler |
| Llama 3.2 3B | Model used for extraction, search answers, planning, and deduplication |
| SQLite | Conversation buffer and temporary fact staging |
| ChromaDB | Persistent vector collection for semantic memories |
| DuckDuckGo search (`ddgs`) | Online text search |
| Trafilatura | Fetching and extracting text from pages in deep-search mode |

Python dependencies are listed in `requirements.txt`: `streamlit`,
`chromadb`, `ollama`, `duckduckgo-search`, and `trafilatura`. The setup script
pulls the two Ollama models used by the project.

## 4. Architecture and data flow

### 4.1 User interfaces

#### Streamlit interface

`src/app.py` implements the browser interface. It maintains the active
conversation ID, database connection, visible messages, and selected
interaction mode in `st.session_state`. Users can choose among Chat, Search,
Deepsearch, Remember, and Incognito.

For regular modes, the app writes the user's turn to SQLite before invoking
the mode handler, then writes the assistant turn after it returns. Incognito
messages are not written to the conversation database. A thumbs-down action
deletes the stored user and assistant rows for that answer, preventing those
turns from reaching the later consolidation step.

The “End Session & Save Memory” button invokes consolidation and then
deduplication. The app displays per-turn buffer-write and inference timings.

#### Command-line interface

`src/chat.py` provides a separate terminal chat loop. It generates a session
UUID, collects input, routes prefixes such as `:search:` and `:remember:`,
and asks the user whether each non-incognito interaction should be saved.
Typing `exit` runs consolidation and deduplication before closing the
connection.

The CLI has a query-planning step for `:search:` and `:deepsearch:`: it asks
the local model for a JSON list of search queries, validates that list, and
passes it to the router. The Streamlit handler map currently calls the router
search functions directly, so it searches the original text rather than
calling the CLI's planner helper.

### 4.2 Routing and web research

`src/router.py` maps explicit prefixes to modes:

```python
modes = [(":search:", "search"), (":deepsearch:", "deepsearch"),
         (":remember:", "remember"), (":incognito:", "incognito")]


def route_message(user_input):
    for prefix, mode in modes:
        if user_input.startswith(prefix):
            cleaned_input = user_input.removeprefix(prefix).strip()
            return mode, cleaned_input

    return normal, user_input
```

Standard search sends up to five results per query to the answer model.
Deep search requests up to ten results per query, fetches pages, extracts
their text, and supplies the extracted material to the synthesis prompt.
The search functions accept optional planned queries, which lets the CLI
search multiple complementary queries while preserving the original user
question for answer generation. Duplicate result URLs are skipped in those
handlers.

The search helper uses a 20-second DDGS timeout and retries once by default.
Incognito uses online search as well, but skips the conversation database and
long-term personal-memory retrieval.

### 4.3 Short-term conversation buffer

The conversation buffer is stored in `data/memory.db`. Its main table has
conversation role and content, a session identifier, a timestamp, and a
consolidation flag. The primary table definitions are:

```sql
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    consolidated INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_text TEXT NOT NULL,
    fact_hash TEXT UNIQUE NOT NULL,
    embedding_id TEXT,
    consolidated_date DATE DEFAULT CURRENT_DATE,
    source_session_id TEXT,
    synced_to_chroma INTEGER DEFAULT 0
);
```

`facts` acts as a staging table during consolidation and deduplication. In
`src/dedup.py`, facts marked synced are deleted from SQLite after processing;
the long-term semantic documents are maintained in ChromaDB.

### 4.4 Consolidation and fact extraction

`src/consolidate.py` reads up to 50 unconsolidated rows for a session, formats
them as a role-labelled transcript, and asks the local LLM to extract
high-value user facts. The prompt requires facts to be explicit, durable, and
attributable to the user. If the model returns `NO_FACTS`, the turns are
marked consolidated without creating facts.

Extracted facts are SHA-256 hashed and inserted into SQLite, skipping exact
hash duplicates. Successful consolidation marks the selected conversation
rows as consolidated and deletes those raw rows when facts were saved.

Core extraction and hashing logic:

```python
def get_facts(transcript):
    prompt = prompts.get_facts_prompt(transcript)
    facts = ollama.chat(
        model=MODEL_EXTRACTION,
        messages=[{"role": "system", "content": prompt}],
    )
    reply = facts["message"]["content"]

    if reply.strip() == "NO_FACTS":
        return None

    lines = reply.split("\n")
    return [line.lstrip("- ").strip()
            for line in lines if line.strip()]


def hash_converter(fact_list):
    fact_data = []
    for line in fact_list:
        hash_value = hashlib.sha256(line.encode()).hexdigest()
        fact_data.append((line, hash_value))
    return fact_data
```

### 4.5 Semantic deduplication and long-term memory

`src/dedup.py` creates or opens a persistent ChromaDB collection named
`main`, configured for cosine distance. For every staged fact, the code
queries the closest existing document. It converts cosine distance to a
similarity score using `1 - distance`.

The three-band thresholds are:

```python
AUTO_DUPLICATE = 0.90
AUTO_DISTINCT = 0.70
```

Decision flow:

1. If ChromaDB has no existing fact, classify the new fact as distinct.
2. Similarity of at least `0.90`: automatically treat the fact as a duplicate
   and skip adding it.
3. Similarity below `0.70`: automatically add it as distinct.
4. Similarity between `0.70` and `0.90`: ask the local LLM whether the facts
   are `DUPLICATE`, `UPDATE`, or `DISTINCT`.
5. For `UPDATE`, delete the matched Chroma document and add the new fact.
   For `DISTINCT`, add the new fact.
6. Mark each staged SQLite fact synced, then clear synced rows when processing
   completes.

The decision mechanism is represented in code as:

```python
if similarity is None:
    decision = "DISTINCT"
elif similarity >= AUTO_DUPLICATE:
    decision = "DUPLICATE"
elif similarity < AUTO_DISTINCT:
    decision = "DISTINCT"
else:
    decision = resolve_conflict(matched_text, fact_text)
```

Memory recall queries ChromaDB for the five nearest documents, then supplies
those facts and the user's question to a grounded response prompt. If no
results are returned, the router responds that no relevant memories are
available.

### 4.6 Prompt design

`src/prompts.py` centralizes prompts for:

- Grounded web-search answers.
- Deep-search research synthesis.
- Incognito web-grounded answers.
- Durable fact extraction.
- Memory-based responses.
- Duplicate/update/distinct classification.
- JSON query planning.

The web-answer prompts were revised to request a direct conclusion and a
reasoned synthesis instead of a source-by-source recap. They ask the model to
avoid repetitive phrases about “search results,” give practical
recommendations when supported, explain trade-offs, and state material
limitations only once. Retrieved webpage text and memory are explicitly
treated as data, not instructions.

The CLI planner's JSON output is parsed and validated before use:

```python
plan = json.loads(content)
if not isinstance(plan, dict):
    raise ValueError("The model search plan must be a JSON object.")

search_queries = plan.get("search_queries")
if not isinstance(search_queries, list) or not all(
    isinstance(query, str) for query in search_queries
):
    raise ValueError(
        "The model search plan must contain a list of search queries."
    )
```

The current planner helper is used in the CLI search handlers. Streamlit
search and deep-search use the router handlers directly and do not call this
planner helper.

## 5. Main interaction modes

| Mode | Purpose | Personal memory behavior | Internet behavior |
|---|---|---|---|
| Chat | Local LLM conversation using stored session history | Logged to SQLite; can be consolidated | No web search |
| Search | Search results synthesized into an answer | Logged in normal app/CLI flow | DDGS web search |
| Deepsearch | Page extraction followed by research synthesis | Logged in normal app/CLI flow | DDGS plus page fetching |
| Remember | Answer using semantically retrieved stored facts | Reads long-term ChromaDB memories; interaction is normally logged | No web search |
| Incognito | Non-memory web-grounded answer | Does not use prior memories or save turns | DDGS web search |

In the CLI, users enter modes by typing prefixes. In Streamlit, users select a
mode with buttons; manually entered prefixes are also routed.

## 6. Benchmarking and test artifacts

`src/benchmark.py` compares the production deduplication decision path with
an exact-string-match baseline. Its hand-labelled set contains 28 pairs:
12 paraphrased duplicates, 12 easy non-duplicates, and 4 hard non-duplicates
with similar subject matter but a changed value or stance. It calculates
precision, recall, accuracy, and F1, prints per-pair decisions, and writes
`benchmark_results.json` and `benchmark_results.png`.

The currently checked-in `benchmark_results.json` contains these results:

| Metric | Exact-match baseline | Three-band system |
|---|---:|---:|
| Precision | 0.000 | 0.778 |
| Recall | 0.000 | 0.583 |
| Accuracy | 0.571 | 0.750 |
| F1 | 0.000 | 0.667 |

These are stored benchmark results, not a claim that the benchmark was
re-run for this report. The evaluation set is small and hand-labelled, so
the figures should be treated as an initial project evaluation, not broad
evidence of production-level accuracy.

The `tests` folder currently contains manual/demo scripts: a simple Ollama
request, a ChromaDB query example, and a SQLite setup/deletion script. They
are not a comprehensive automated unit-test suite.

## 7. Running the application

The provided setup script is intended to:

1. Create a `.venv` virtual environment.
2. Install packages from `requirements.txt`.
3. check that Ollama is available.
4. Pull `qwen2.5:1.5b` and `llama3.2:3b`.

The README describes launching Streamlit with the supplied run script. There
is a path mismatch to resolve before relying on that launcher: the tracked
Streamlit file is `src/app.py`, while `run.bat` currently invokes
`streamlit run app.py` from the repository root. Unless a root-level
`app.py` is created or the script is adjusted, the command will not point to
the tracked application file.

The default SQLite path is `data/memory.db`; the ChromaDB persistent client
uses the default local persistence directory. The actual tracked/runtime
state may vary by installation.

## 8. Privacy and deployment considerations

Local Ollama inference and local SQLite/ChromaDB persistence support the
project's local-first goal. The web-search functionality is a separate
networked feature:

- Search queries are sent to the DuckDuckGo search service through the
  `ddgs` client.
- Deep search downloads selected webpages using Trafilatura.
- Therefore, web-search queries and requests are not confined to the device.
- Incognito prevents this application's conversation persistence and memory
  recall; it does not make online search traffic offline or anonymous.

The README's broad statement that “nothing leaves this laptop” should be
qualified wherever web-search modes are enabled.

## 9. Current implementation observations

These observations describe the checked-in source and are useful context for
presentations or future development:

1. **Two entry points have different behavior.** The Streamlit app is the
   documented primary UI. The CLI has its own search-planning functions.
   Streamlit currently invokes router search handlers directly, bypassing
   query planning.
2. **CLI chat does not include the current input in model history.** The CLI
   calls the chat handler before saving the current user turn, and
   `handle_chat` builds its model messages from saved history without adding
   its `cleaned_input` parameter. This means the current question is not
   included in the chat model request. The Streamlit handler saves the user
   turn before it loads history, so this behavior differs between interfaces.
3. **Model selection differs by task and interface.** Streamlit's chat
   handler selects Qwen 2.5 1.5B. CLI chat and many router tasks use Llama
   3.2 3B; deduplication also selects Llama 3.2 3B.
4. **The setup/run paths need alignment.** `setup.bat` creates `.venv`,
   while `run.bat` starts `app.py` at the repository root; the tracked app is
   under `src`.
5. **The project status is inconsistent in the README.** It labels the
   project “In Development” near the top and “Completed” at the end.
6. **The benchmark is an initial evaluation.** It uses 28 curated examples,
   with the stored result showing improved F1 and accuracy over exact match,
   but leaves room to improve recall and to evaluate on a larger dataset.
7. **The README and network behavior should be reconciled.** Core model
   inference and memory are local, but internet search and page fetches are
   network-dependent.
8. **Automated regression coverage is limited.** Existing test scripts are
   demonstrations rather than focused tests for consolidation, the
   three-band deduplication path, session feedback, or planner validation.

## 10. Potential next steps

- Fix and verify the `run.bat` application path and document a reliable
  manual launch command.
- Share the query-planning behavior between CLI and Streamlit so both
  interfaces produce consistent search results.
- Clearly disclose that web-search modes make external network requests.
- Add unit tests around SQLite consolidation, exact-hash duplicates, each
  similarity band, LLM conflict labels, malformed planner JSON, and
  incognito persistence boundaries.
- Expand the deduplication benchmark with more examples, broader paraphrase
  patterns, explicit score reporting, and repeated runs for ambiguous cases.
- Measure model latency, memory growth, and search quality on representative
  hardware and realistic interactions.
- Consider storing or surfacing source URLs in generated answers in a
  consistent user-facing citation format.

## 11. Conclusion

The project demonstrates a practical local-first assistant architecture that
separates short-term conversation history from long-term semantic memory. Its
most substantial component is the consolidation and deduplication pipeline:
the system extracts explicit facts, filters exact repeats, applies vector
similarity thresholds, and reserves LLM reasoning for borderline cases.
Streamlit and CLI interfaces provide several ways to interact with this
memory and with external web research.

The stored benchmark indicates that semantic comparison improves on exact
matching for the current curated examples, though recall and evaluation scale
remain areas for improvement. The strongest next steps are to align the
interfaces and launch scripts, increase automated testing, and describe the
network implications of web search precisely.

## Source file guide

- `src/app.py` — Streamlit interface and interaction lifecycle.
- `src/chat.py` — Command-line chat, persistence choices, and CLI query
  planning.
- `src/router.py` — Routing, web research, answer handlers, and memory recall.
- `src/consolidate.py` — Session transcript processing and fact extraction.
- `src/dedup.py` — ChromaDB similarity checks and three-band deduplication.
- `src/prompts.py` — Model prompt templates.
- `src/benchmark.py` — Curated deduplication benchmark and metric generation.
- `data/schema.sql` — SQLite schema for conversations and fact staging.
- `requirements.txt` — Python package dependencies.
- `setup.bat`, `run.bat` — Windows setup and launch scripts.
- `benchmark_results.json`, `benchmark_results.png` — Stored benchmark
  outputs.
