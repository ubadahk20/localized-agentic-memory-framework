"""
Module 6 - Deduplication Benchmark (v2 - tests the real 3-layer pipeline)

Task: given a pair of facts, predict whether they are duplicates.

Compared against hand-labeled ground truth:
  - Baseline  : exact string match only (no semantic dedup at all)
  - Our system: the REAL dedup.py pipeline - hash match, then cosine
                similarity with a 3-way band (auto-duplicate / ambiguous ->
                LLM decides / auto-distinct), same code path production uses.

Run from the project root:   python src/benchmark.py
Needs Ollama running locally with the model set in dedup.py.
"""

import dedup  # the real production module - this is the whole point of v2
import matplotlib.pyplot as plt
import hashlib
import json

import matplotlib
matplotlib.use("Agg")


DUPLICATES = [
    ("User's name is Ubadah", "The user is called Ubadah"),
    ("User works as a Data Engineer", "The user is employed as a Data Engineer"),
    ("User is 21 years old", "The user's age is 21"),
    ("User enjoys playing cricket", "The user likes cricket"),
    ("User is a final year IT student",
     "The user is studying Information Technology in their final year"),
    ("User prefers direct feedback", "The user likes honest, straightforward feedback"),
    ("User is interested in AI/ML roles",
     "The user wants a career in artificial intelligence and machine learning"),
    ("User's project deadline is mid-September",
     "The user needs to submit their project by mid-September"),
    ("User is based in Navi Mumbai", "The user lives in Navi Mumbai"),
    ("User runs a mango business", "The user is involved in selling mangoes seasonally"),
    ("User wants to relocate to Bangalore",
     "The user is open to moving to Bangalore for work"),
    ("User completed a diploma in Computer Engineering",
     "The user holds a diploma in Computer Engineering"),
]

EASY_NEGATIVES = [
    ("User's name is Ubadah", "User enjoys playing cricket"),
    ("User works as a Data Engineer", "User's project deadline is mid-September"),
    ("User is 21 years old", "User runs a mango business"),
    ("User is interested in AI/ML roles", "User is based in Navi Mumbai"),
    ("User prefers direct feedback",
     "User completed a diploma in Computer Engineering"),
    ("User is a final year IT student", "User wants to relocate to Bangalore"),
    ("Creatine is an amino acid naturally present in the body",
     "The best creatine supplement is Thorne Creatine Powder"),
    ("Imam Abu Hanifa founded the Hanafi school of jurisprudence",
     "The Shafi'i school is named after al-Shafi'i"),
    ("The demand for AI/ML engineers in India is growing rapidly",
     "NASSCOM estimates India will need 1 million AI-skilled professionals by 2027"),
    ("James Bond was created by Ian Fleming in 1953",
     "Daniel Craig played James Bond in No Time to Die"),
    ("Blockchain is decentralized and distributed",
     "Smart contracts automate agreements without intermediaries"),
    ("Malaysia's capital is Kuala Lumpur",
     "Singapore is one of the easiest countries to move to from India"),
]

# These are NOT duplicates - same topic, one key detail differs. This is
# exactly what Layer 3 (the LLM conflict check) exists to catch correctly,
# where pure cosine similarity was getting these wrong (see v1 results).
HARD_NEGATIVES = [
    ("User is 21 years old", "User is 22 years old"),
    ("User works at Delhivery", "User works at Zomato"),
    ("User likes cricket", "User dislikes cricket"),
    ("User's project deadline is September 18",
     "User's project deadline is September 25"),
]

TEST_PAIRS = (
    [(a, b, True, "duplicate") for a, b in DUPLICATES]
    + [(a, b, False, "easy negative") for a, b in EASY_NEGATIVES]
    + [(a, b, False, "hard negative") for a, b in HARD_NEGATIVES]
)


def norm(s):
    return s.strip().lower()


def baseline_predict(a, b):
    return norm(a) == norm(b)


def system_predict_v2(a, b, collection):
    """Runs the EXACT same decision path as dedup.py's trigger_deduplication:
    hash check, then cosine similarity, then the 3-way band (auto-dup /
    LLM-resolved / auto-distinct). 'Duplicate' here means anything that
    would NOT be added as a brand-new fact (i.e. DUPLICATE or UPDATE both
    count as "matched to something existing" for this pair-classification task)."""
    hash_a = hashlib.sha256(norm(a).encode()).hexdigest()
    hash_b = hashlib.sha256(norm(b).encode()).hexdigest()
    if hash_a == hash_b:
        return True, "hash"

    similarity, matched_text, matched_id = dedup.check_similarity(
        collection, b)
    # matched_text should be `a` here since we add `a` fresh just before calling this

    if similarity is None:
        return False, "empty"
    elif similarity >= dedup.AUTO_DUPLICATE:
        return True, f"auto_dup(sim={similarity:.3f})"
    elif similarity < dedup.AUTO_DISTINCT:
        return False, f"auto_distinct(sim={similarity:.3f})"
    else:
        decision = dedup.resolve_conflict(matched_text, b)
        is_dup = decision in ("DUPLICATE", "UPDATE")
        return is_dup, f"llm={decision}(sim={similarity:.3f})"


def metrics(preds, truth):
    tp = sum(p and t for p, t in zip(preds, truth))
    fp = sum(p and not t for p, t in zip(preds, truth))
    fn = sum((not p) and t for p, t in zip(preds, truth))
    tn = sum((not p) and (not t) for p, t in zip(preds, truth))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(preds)
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1}


def make_chart(base, system, path):
    names = ["precision", "recall", "accuracy", "f1"]
    labels = ["Precision", "Recall", "Accuracy", "F1"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(names))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], [base[n] for n in names], w,
                label="Baseline (exact match)", color="#9AA3AD")
    b2 = ax.bar([i + w / 2 for i in x], [system[n] for n in names], w,
                label="Our system (3-layer dedup)", color="#3B7A57")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_title("Duplicate detection: baseline vs 3-layer system")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import chromadb
    client = chromadb.Client()  # fresh in-memory collection, isolated from the real store

    truth = [p[2] for p in TEST_PAIRS]
    base_preds = [baseline_predict(a, b) for a, b, _, _ in TEST_PAIRS]

    sys_preds = []
    reasons = []
    for i, (a, b, _, _) in enumerate(TEST_PAIRS):
        col = client.create_collection(name=f"bench_{i}", metadata={
                                       "hnsw:space": "cosine"})
        col.add(documents=[a], ids=["a"])
        pred, reason = system_predict_v2(a, b, col)
        sys_preds.append(pred)
        reasons.append(reason)
        client.delete_collection(f"bench_{i}")

    base_m = metrics(base_preds, truth)
    sys_m = metrics(sys_preds, truth)

    print("=" * 66)
    print(f"{'Metric':<12}{'Baseline (exact match)':<26}{'Our system (3-layer)'}")
    print("=" * 66)
    for n in ["precision", "recall", "accuracy", "f1"]:
        print(f"{n:<12}{base_m[n]:<26.3f}{sys_m[n]:.3f}")
    print("-" * 66)
    print("Baseline:", {k: base_m[k] for k in ("TP", "FP", "FN", "TN")})
    print("System  :", {k: sys_m[k] for k in ("TP", "FP", "FN", "TN")})

    print("\nAll pairs with how the system decided:")
    for (a, b, t, cat), pred, reason in zip(TEST_PAIRS, sys_preds, reasons):
        mark = "OK " if pred == t else "ERR"
        print(f"  [{mark}][{cat}] {reason} | predicted={'dup' if pred else 'not dup'} "
              f"truth={'dup' if t else 'not dup'}")
        print(f"        '{a}' vs '{b}'")

    make_chart(base_m, sys_m, "benchmark_results.png")
    with open("benchmark_results.json", "w") as f:
        json.dump({"baseline": base_m, "system": sys_m}, f, indent=2)
    print("\nSaved benchmark_results.png and benchmark_results.json")
