import trafilatura
from ddgs import DDGS


def handle_deepsearch(cleaned_input):
    cleaned_result = []
    with DDGS() as ddgs:
        for r in ddgs.text(cleaned_input, max_results=3):
            url = r["href"]
            downloaded = trafilatura.fetch_url(url)
            result = trafilatura.extract(downloaded)
            cleaned_result.append(f"{url} : {result}")
    formatted_result = "\n".join(cleaned_result)
   # response = ollama.chat(model)


handle_deepsearch("current UPI fraud regulations")
