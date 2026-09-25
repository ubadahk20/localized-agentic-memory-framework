import chromadb
import sqlite3


def get_chroma_collection():
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name="main", metadata={"hnsw:space": "cosine"})
    return collection


collection = get_chroma_collection()


def get_new_facts(conn):
    documents = []
    rows = conn.execute(
        "SELECT id, fact_text FROM facts WHERE synced_to_chroma = 0").fetchall()
    str_ids = [str(row[0]) for row in rows]
    documents = [row[1] for row in rows]
    return str_ids, documents


def sync_new_facts_to_chroma(conn, collection, str_ids, documents):

    if str_ids != []:
        collection.add(documents=documents, ids=str_ids)
        for id in str_ids:
            conn.execute(
                "UPDATE facts SET synced_to_chroma = 1 WHERE id = ?", (id,))
        conn.commit()
        return documents

    else:

        return None


def check_similarity(collection, new_fact_test):
    result = collection.query(
        query_texts=[new_fact_test], n_results=1)
    if not result["ids"][0]:
        return False, None, None
    distances = result["distances"][0]
    closest_distance = distances[0]
    matched_text = result["documents"][0][0]
    similarity = 1 - closest_distance
    print(similarity)
    is_duplicate = similarity >= 0.85
    return is_duplicate, matched_text, similarity


def clear_synced_facts(conn):
    # Delete rows from `facts` where synced_to_chroma = 1 —
    conn.execute("DELETE FROM facts WHERE synced_to_chroma = 1")
    conn.commit()
    # they're safely persisted in Chroma now, no need to keep them in SQLite


def trigger_deduplication(conn):
    str_ids, documents = get_new_facts(conn)

    if documents:
        for fact_id, fact_text in zip(str_ids, documents):
            # 1. Check similarity first against what's already in Chroma
            is_dup, matched, score = check_similarity(collection, fact_text)

            if is_dup:
                print(
                    f"Duplicate found! Skipping: '{fact_text}' (Matched: '{matched}', Score: {score})")
            # Optionally mark it as synced or handled in SQLite so it doesn't keep pulling
                conn.execute(
                    "UPDATE facts SET synced_to_chroma = 1 WHERE id = ?", (fact_id,))
                conn.commit()
            else:
                print(f"Unique fact. Adding to Chroma: '{fact_text}'")
            # Add them individually or collect unique ones to add in bulk
                collection.add(documents=[fact_text], ids=[fact_id])
                conn.execute(
                    "UPDATE facts SET synced_to_chroma = 1 WHERE id = ?", (fact_id,))
                conn.commit()
                clear_synced_facts(conn)

    else:
        print("no new facts")


if __name__ == "__main__":
    trigger_deduplication()
