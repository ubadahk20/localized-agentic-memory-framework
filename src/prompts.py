import chromadb
import sqlite3
import ollama
import prompts

MODEL = "llama3.2:3b"

# Three-band thresholds — avoids burning an LLM call on every fact.
# Only the ambiguous middle band gets real reasoning.
AUTO_DUPLICATE = 0.90   # above this: confident enough to skip the LLM call
AUTO_DISTINCT = 0.70    # below this: confident enough it's unrelated


def get_chroma_collection():
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name="main", metadata={"hnsw:space": "cosine"})
    return collection


collection = get_chroma_collection()


def get_new_facts(conn):
    rows = conn.execute(
        "SELECT id, fact_text FROM facts WHERE synced_to_chroma = 0").fetchall()
    str_ids = [str(row[0]) for row in rows]
    documents = [row[1] for row in rows]
    return str_ids, documents


def check_similarity(collection, new_fact_text):
    """Returns (similarity, matched_text, matched_id) for the closest existing
    fact, or (None, None, None) if the collection is empty."""
    result = collection.query(query_texts=[new_fact_text], n_results=1)
    if not result["ids"][0]:
        return None, None, None
    similarity = 1 - result["distances"][0][0]
    matched_text = result["documents"][0][0]
    matched_id = result["ids"][0][0]
    return similarity, matched_text, matched_id


def resolve_conflict(existing_fact, new_fact):
    """Only called for the ambiguous middle band. Asks the LLM whether the
    two facts are the same thing, an update/correction, or genuinely
    different information."""
    prompt = prompts.get_conflict_resolution_prompt(existing_fact, new_fact)
    response = ollama.chat(model=MODEL, messages=[
                           {"role": "user", "content": prompt}])
    decision = response["message"]["content"].strip().upper()

    if "DUPLICATE" in decision:
        return "DUPLICATE"
    elif "UPDATE" in decision:
        return "UPDATE"
    else:
        return "DISTINCT"


def clear_synced_facts(conn):
    conn.execute("DELETE FROM facts WHERE synced_to_chroma = 1")
    conn.commit()


def trigger_deduplication(conn):
    str_ids, documents = get_new_facts(conn)

    if not documents:
        print("no new facts")
        return

    for fact_id, fact_text in zip(str_ids, documents):
        similarity, matched_text, matched_id = check_similarity(
            collection, fact_text)

        if similarity is None:
            # Empty collection — nothing to compare against, definitely new
            decision = "DISTINCT"
        elif similarity >= AUTO_DUPLICATE:
            decision = "DUPLICATE"
        elif similarity < AUTO_DISTINCT:
            decision = "DISTINCT"
        else:
            # Ambiguous band — ask the LLM to actually reason about it
            decision = resolve_conflict(matched_text, fact_text)
            print(f"Ambiguous (sim={similarity:.3f}) — LLM decided: {decision} "
                  f"| existing: '{matched_text}' | new: '{fact_text}'")

        if decision == "DUPLICATE":
            print(f"Duplicate. Skipping: '{fact_text}'")

        elif decision == "UPDATE":
            print(f"Update. Replacing '{matched_text}' with '{fact_text}'")
            collection.delete(ids=[matched_id])
            collection.add(documents=[fact_text], ids=[fact_id])

        else:  # DISTINCT
            print(f"New fact. Adding: '{fact_text}'")
            collection.add(documents=[fact_text], ids=[fact_id])

        conn.execute(
            "UPDATE facts SET synced_to_chroma = 1 WHERE id = ?", (fact_id,))
        conn.commit()

    clear_synced_facts(conn)


if __name__ == "__main__":
    trigger_deduplication()
