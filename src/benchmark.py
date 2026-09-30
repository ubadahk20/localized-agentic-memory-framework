"""
Module 6 - Deduplication Benchmark

Task: given a pair of facts, predict whether they are duplicates (same
underlying information, possibly worded differently).

Compared against a hand-labeled ground truth:
  - Baseline  : exact string match only. This is a stand-in for systems with
                no semantic dedup (e.g. plain RAG pipelines). It is NOT a
                reproduction of any reviewed paper's system.
  - Our system: SHA-256 exact match OR ChromaDB cosine similarity >= threshold.

Run from the project root:   python src/benchmark.py
Needs:                       pip install matplotlib
"""

import matplotlib.pyplot as plt
import hashlib
import json

import chromadb
import matplotlib
matplotlib.use("Agg")

PRODUCTION_THRESHOLD = 0.85   # must match the value used in dedup.py
SWEEP = [round(0.50 + 0.05 * i, 2) for i in range(10)]   # 0.50 ... 0.95

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

# Near-identical wording but a key detail differs, so these are NOT duplicates.
# Pure similarity dedup is expected to struggle here (this is what a Layer 3
# LLM check would be for).
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


def compute_similarities(pairs):
    """Cosine similarity per pair, using a fresh isolated collection each time
    so a fact from one pair can never match a fact from another pair."""
    client = chromadb.Client()   # in-memory, does not touch the real store
    sims = []
    for i, (a, b, _, _) in enumerate(pairs):
        name = f"bench_{i}"
        col = client.create_collection(
            name=name, metadata={"hnsw:space": "cosine"})
        col.add(documents=[a], ids=["a"])
        res = col.query(query_texts=[b], n_results=1)
        sims.append(1 - res["distances"][0][0])
        client.delete_collection(name)
    return sims


def norm(s):
    return s.strip().lower()


def baseline_predict(a, b):
    return norm(a) == norm(b)


def system_predict(a, b, sim, threshold):
    same_hash = hashlib.sha256(norm(a).encode()).hexdigest(
    ) == hashlib.sha256(norm(b).encode()).hexdigest()
    return same_hash or sim >= threshold


def metrics(preds, truth):
    tp = sum(p and t for p, t in zip(preds, truth))
    fp = sum(p and not t for p, t in zip(preds, truth))
    fn = sum((not p) and t for p, t in zip(preds, truth))
    tn = sum((not p) and (not t) for p, t in zip(preds, truth))
    # 0 when nothing is predicted positive
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(preds)
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1}


def evaluate(pairs, sims):
    truth = [p[2] for p in pairs]
    base = metrics([baseline_predict(a, b) for a, b, _, _ in pairs], truth)
    sweep = {}
    for th in SWEEP:
        preds = [system_predict(a, b, s, th)
                 for (a, b, _, _), s in zip(pairs, sims)]
        sweep[th] = metrics(preds, truth)
    return {"baseline": base, "system": sweep[PRODUCTION_THRESHOLD], "sweep": sweep}


def make_charts(results, path):
    base, system, sweep = results["baseline"], results["system"], results["sweep"]
    names = ["precision", "recall", "accuracy", "f1"]
    labels = ["Precision", "Recall", "Accuracy", "F1"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = range(len(names))
    w = 0.38
    b1 = ax1.bar([i - w / 2 for i in x], [base[n] for n in names], w,
                 label="Baseline (exact match)", color="#9AA3AD")
    b2 = ax1.bar([i + w / 2 for i in x], [system[n] for n in names], w,
                 label=f"Our system (hash + cosine >= {PRODUCTION_THRESHOLD})", color="#3B7A57")
    for bars in (b1, b2):
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                     f"{bar.get_height():.2f}", ha="center", fontsize=9)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.12)
    ax1.set_title("Duplicate detection: baseline vs our system")
    ax1.legend(loc="upper left", fontsize=9)

    ths = list(sweep.keys())
    for n, lab, col in [("precision", "Precision", "#B0413E"),
                        ("recall", "Recall", "#3B7A57"),
                        ("f1", "F1", "#1F2937")]:
        ax2.plot(ths, [sweep[t][n]
                 for t in ths], marker="o", label=lab, color=col)
    ax2.axvline(PRODUCTION_THRESHOLD, linestyle="--", color="#9AA3AD")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Cosine similarity threshold")
    ax2.set_title(
        "Threshold sensitivity (dashed = threshold used in dedup.py)")
    ax2.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def report(pairs, sims, results):
    base, system = results["baseline"], results["system"]
    print("=" * 66)
    print(f"{'Metric':<12}{'Baseline (exact match)':<26}{'Our system @ ' + str(PRODUCTION_THRESHOLD)}")
    print("=" * 66)
    for n in ["precision", "recall", "accuracy", "f1"]:
        print(f"{n:<12}{base[n]:<26.3f}{system[n]:.3f}")
    print("-" * 66)
    print("Baseline :", {k: base[k] for k in ("TP", "FP", "FN", "TN")})
    print("System   :", {k: system[k] for k in ("TP", "FP", "FN", "TN")})
    print("\nPairs our system got wrong at the production threshold:")
    for (a, b, truth, cat), s in zip(pairs, sims):
        pred = system_predict(a, b, s, PRODUCTION_THRESHOLD)
        if pred != truth:
            print(
                f"  [{cat}] sim={s:.3f} predicted={'dup' if pred else 'not dup'} | {a!r} vs {b!r}")


if __name__ == "__main__":
    sims = compute_similarities(TEST_PAIRS)
    results = evaluate(TEST_PAIRS, sims)
    report(TEST_PAIRS, sims, results)
    make_charts(results, "benchmark_results.png")
    with open("benchmark_results.json", "w") as f:
        json.dump({"threshold": PRODUCTION_THRESHOLD,
                   "baseline": results["baseline"],
                   "system": results["system"],
                   "sweep": {str(k): v for k, v in results["sweep"].items()}}, f, indent=2)
    print("\nSaved benchmark_results.png and benchmark_results.json")
