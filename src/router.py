import prompts
from ddgs import DDGS
import ollama
import trafilatura


MODEL = "qwen2.5:1.5b"

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

# handle_search() is a special handler that fetches the top 10 search
# results and passes them to the model for processing


def handle_search(cleaned_input):
    cleaned_result = []

    with DDGS() as ddgs:
        for r in ddgs.text(cleaned_input, max_results=10):
            title = r["title"]
            body = r["body"]
            cleaned_result.append(f"{title}: {body}")
    formatted_result = "\n".join(cleaned_result)
    prompt = prompts.get_search_prompt(cleaned_input, formatted_result)
    response = ollama.chat(model=MODEL, messages=[{"role": "user",
                                                   "content": prompt}])

    reply = response["message"]["content"]
    return reply

# handle_deepsearch() is a special handler that fetches the full text of
# the top 10 search results and passes them to the model for processing


def handle_deepsearch(cleaned_input):
    cleaned_result = []
    with DDGS() as ddgs:
        for r in ddgs.text(cleaned_input, max_results=10):
            url = r["href"]
            downloaded = trafilatura.fetch_url(url)
            result = trafilatura.extract(downloaded)
            if result is not None:
                shortened_text = result[:1000]
                cleaned_result.append(f"{shortened_text}")
    formatted_result = "\n".join(cleaned_result)
    prompt = prompts.get_deepsearch_prompt(cleaned_input, formatted_result)
    response = ollama.chat(model=MODEL, messages=[
                           {"role": "user", "content": prompt}])

    reply = response["message"]["content"]
    return reply


def handle_remember(cleaned_input):
    return f"[No long-term memory system yet — this will query ChromaDB once Module 5 is built. Your query was: '{cleaned_input}']"

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
    response = ollama.chat(model=MODEL, messages=[{"role": "user",
                                                   "content": prompt}])

    reply = response["message"]["content"]
    return reply
