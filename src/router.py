import prompts
from ddgs import DDGS
import ollama
import trafilatura
import dedup


MODEL_FAST = "qwen2.5:1.5b"
MODEL_EXTRACTION = "llama3.2:3b"

# load_history() is a database retriever


def load_history(conn, session_id):
    rows = conn.execute(
        "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id",
        (session_id,)
    ).fetchall()
    return [{'role': r,  'content': c} for r, c in rows]


modes = [(":search:", "search"), (":deepsearch:", "deepsearch"),
         (":remember:", "remember"), (":incognito:", "incognito")]
normal = "chat"


# route_message() tells us which mode to use based on the user input prefix

def route_message(user_input):
    for prefix, mode in modes:
        if user_input.startswith(prefix):
            cleaned_input = user_input.removeprefix(prefix).strip()
            return mode, cleaned_input

    return normal, user_input

# Replace handle_search() and handle_deepsearch() in router.py with these two versions.
# What changed: DDGS(timeout=20) gives it more time on a slow connection instead of
# the library's short default, and a simple retry-once wrapper absorbs one-off
# network blips (the Brave/Google fallback timeouts you saw) instead of failing
# the whole request on the first hiccup.


def _search_with_retry(query, max_results, attempts=2):
    """Runs a DDGS text search with a longer timeout and one retry on failure."""
    last_error = None
    for attempt in range(attempts):
        try:
            with DDGS(timeout=20) as ddgs:
                return list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            last_error = e
    raise last_error


def handle_search(cleaned_input):
    cleaned_result = []
    results = _search_with_retry(cleaned_input, max_results=10)
    for r in results:
        title = r["title"]
        body = r["body"]
        cleaned_result.append(f"{title}: {body}")
    formatted_result = "\n".join(cleaned_result)
    prompt = prompts.get_search_prompt(cleaned_input, formatted_result)
    response = ollama.chat(model=MODEL_EXTRACTION, messages=[
                           {"role": "user", "content": prompt}])
    reply = response["message"]["content"]
    return reply


def handle_deepsearch(cleaned_input):
    cleaned_result = []
    results = _search_with_retry(cleaned_input, max_results=10)
    for r in results:
        url = r["href"]
        downloaded = trafilatura.fetch_url(url)
        result = trafilatura.extract(downloaded)
        if result is not None:
            shortened_text = result[:1000]
            cleaned_result.append(f"{shortened_text}")
    formatted_result = "\n".join(cleaned_result)
    prompt = prompts.get_deepsearch_prompt(cleaned_input, formatted_result)
    response = ollama.chat(model=MODEL_EXTRACTION, messages=[
                           {"role": "user", "content": prompt}])
    reply = response["message"]["content"]
    return reply
# handle_remember remembers from the vector databse


def handle_remember(cleaned_input):
    result = dedup.collection.query(query_texts=[cleaned_input], n_results=5)

    if not result["ids"][0]:
        return "I don't have any relevant memories about that yet."

    retrieved_facts = "\n".join(result["documents"][0])
    prompt = prompts.get_remember_prompt(cleaned_input, retrieved_facts)
    response = ollama.chat(model=MODEL_EXTRACTION, messages=[
                           {"role": "user", "content": prompt}])
    reply = response["message"]["content"]
    return f"{reply}\n\n[Remembered: {retrieved_facts}]"


# handle_incognito() is a special handler that does not log the conversation to the database


def handle_incognito(cleaned_input):
    cleaned_result = []

    with DDGS() as ddgs:
        for r in ddgs.text(cleaned_input, max_results=10):
            title = r["title"]
            body = r["body"]
            cleaned_result.append(f"{title}: {body}")
    formatted_result = "\n".join(cleaned_result)
    prompt = prompts.get_incognito_prompt(cleaned_input)
    response = ollama.chat(model=MODEL_EXTRACTION, messages=[{"role": "user",
                                                              "content": prompt}])

    reply = response["message"]["content"]
    return reply
