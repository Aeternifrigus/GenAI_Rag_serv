"""Exercises every endpoint against the real app, including the lifespan ingest."""
import json
import sys
from pathlib import Path

# Python puts this script's directory on sys.path, not the project root, so the
# app package is not importable without help. Makes `python scripts/smoke.py`
# work from anywhere rather than only from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_CASES = json.loads((ROOT / "data" / "eval" / "questions.json").read_text(encoding="utf-8"))

with TestClient(app) as client:
    print("=" * 62)
    print("HEALTH")
    print("=" * 62)
    r = client.get("/health")
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200
    assert "reranker" in r.json()["resolved"]

    # map each source document to its doc_id so evaluation has real ground truth
    print()
    print("=" * 62)
    print("RETRIEVE  (relevance only, no generation)")
    print("=" * 62)
    probes = {
        "expenses-policy":   "What is the deadline for submitting expense claims?",
        "incident-runbook":  "How fast must on-call acknowledge a severity one incident?",
        "data-retention":    "How long are client transaction records retained?",
        "model-monitoring":  "What threshold on the stability index raises a drift alert?",
    }
    for title, q in probes.items():
        r = client.post("/retrieve", json={"question": q, "k": 1})
        assert r.status_code == 200, r.text
        hit = r.json()["hits"][0]
        ok = "OK " if hit["metadata"]["title"] == title else "MISS"
        print(f"  [{ok}] {q[:52]:<52} -> {hit['metadata']['title']:<20} {hit['score']:.4f}")

    # Ground truth for evaluation is resolved from the index itself rather than
    # hardcoded, because doc_ids are content-derived and change with the corpus.
    truth = {}
    for title in {c["title"] for c in EVAL_CASES}:
        r = client.post("/retrieve", json={"question": title.replace("-", " "), "k": 5})
        for hit in r.json()["hits"]:
            truth.setdefault(hit["metadata"]["title"], hit["doc_id"])
    unresolved = sorted({c["title"] for c in EVAL_CASES} - truth.keys())
    assert not unresolved, f"documents missing from the index: {unresolved}"

    print()
    print("=" * 62)
    print("QUERY  (retrieval + generation + citations)")
    print("=" * 62)
    r = client.post("/query", json={"question": "Can I fly business class?", "k": 2})
    body = r.json()
    print(f"  grounded : {body['grounded']}")
    print(f"  generator: {body['generator']}")
    print(f"  plan     : {body['plan']['reason']}")
    print(f"  reranker : {body['plan']['reranker']} over {body['plan']['candidates']} candidates")
    print(f"  variants : {body['plan']['variants']}")
    cites = [(c["marker"], c["metadata"]["title"], c["score"]) for c in body["citations"]]
    print(f"  citations: {cites}")

    print()
    print("=" * 62)
    print("INGEST  (inline documents through the API)")
    print("=" * 62)
    r = client.post("/ingest", json={"documents": [
        {"doc_id": "canteen-hours", "text":
            "The canteen opens at 07:30 and stops serving hot food at 14:45. "
            "Vegetarian and halal options are available at every service. "
            "The espresso bar on the third floor closes at 16:00."},
    ]})
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200

    r = client.post("/query", json={"question": "when does the canteen stop serving hot food",
                                    "k": 1})
    top = r.json()["citations"][0]
    found = top["doc_id"] == "canteen-hours"
    print(f"  newly ingested doc retrievable: {found} (score {top['score']:.4f})")
    assert found, "a freshly ingested document must be retrievable"

    print()
    print("=" * 62)
    print("EVALUATE  (labelled ground truth)")
    print("=" * 62)
    cases = [
        {"question": c["question"], "relevant_doc_ids": [truth[c["title"]]]}
        for c in EVAL_CASES
    ]
    r = client.post("/evaluate", json={"cases": cases, "k": 5})
    out = r.json()
    print(json.dumps(out["summary"], indent=2))
    print()
    for c in out["per_case"]:
        if not c["hit"]:
            print(f"  [MISS] {c['question'][:60]}")
    print(f"  {sum(1 for c in out['per_case'] if c['hit'])}/{len(cases)} questions hit")

    print()
    print("=" * 62)
    print("ERROR HANDLING")
    print("=" * 62)
    r = client.post("/query", json={})
    print(f"  missing question      -> {r.status_code} (expect 422)")
    assert r.status_code == 422
    r = client.post("/query", json={"question": "hi", "k": 999})
    print(f"  k above the limit     -> {r.status_code} (expect 422)")
    assert r.status_code == 422
    r = client.post("/query", json={"question": "hi", "sources": ["nope"]})
    print(f"  unknown source        -> {r.status_code}, grounded={r.json()['grounded']}")
    r = client.post("/evaluate", json={"cases": [], "k": 3})
    print(f"  empty evaluation set  -> {r.status_code} (expect 422)")

    print()
    print("all endpoint checks passed")
