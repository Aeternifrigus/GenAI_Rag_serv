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

with TestClient(app) as client:
    print("=" * 62)
    print("HEALTH")
    print("=" * 62)
    r = client.get("/health")
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200

    # map each source document to its doc_id so evaluation has real ground truth
    print()
    print("=" * 62)
    print("RETRIEVE  (relevance only, no generation)")
    print("=" * 62)
    truth = {}
    probes = {
        "expenses-policy":   "What is the deadline for submitting expense claims?",
        "incident-runbook":  "How fast must on-call acknowledge a severity one incident?",
        "data-retention":    "How long are client transaction records retained?",
        "model-deployment":  "When is a model allowed to be promoted to production?",
    }
    for title, q in probes.items():
        r = client.post("/retrieve", json={"question": q, "k": 1})
        assert r.status_code == 200, r.text
        hit = r.json()["hits"][0]
        truth[title] = hit["doc_id"]
        ok = "OK " if hit["metadata"]["title"] == title else "MISS"
        print(f"  [{ok}] {q[:52]:<52} -> {hit['metadata']['title']:<18} {hit['score']:.4f}")

    print()
    print("=" * 62)
    print("QUERY  (retrieval + generation + citations)")
    print("=" * 62)
    r = client.post("/query", json={"question": "Can I fly business class?", "k": 2})
    body = r.json()
    print(f"  grounded : {body['grounded']}")
    print(f"  generator: {body['generator']}")
    print(f"  plan     : {body['plan']['reason']}")
    print(f"  variants : {body['plan']['variants']}")
    cites = [(c["marker"], c["metadata"]["title"], c["score"]) for c in body["citations"]]
    print(f"  citations: {cites}")

    print()
    print("=" * 62)
    print("INGEST  (inline documents through the API)")
    print("=" * 62)
    r = client.post("/ingest", json={"documents": [
        {"doc_id": "vendor-sla", "text":
            "Vendor support tickets are answered within four business hours for "
            "priority one and within two business days for priority three. "
            "Escalation to a named account manager is available for priority one."},
    ]})
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200

    r = client.post("/query", json={"question": "vendor support ticket response time", "k": 1})
    top = r.json()["citations"][0]
    found = top["doc_id"] == "vendor-sla"
    print(f"  newly ingested doc retrievable: {found} (score {top['score']:.4f})")

    print()
    print("=" * 62)
    print("EVALUATE  (labelled ground truth)")
    print("=" * 62)
    cases = [
        {"question": "What is the deadline for submitting expense claims?",
         "relevant_doc_ids": [truth["expenses-policy"]]},
        {"question": "Do I need a receipt for a 12 EUR expense?",
         "relevant_doc_ids": [truth["expenses-policy"]]},
        {"question": "How fast must on-call acknowledge a severity one incident?",
         "relevant_doc_ids": [truth["incident-runbook"]]},
        {"question": "Should I roll back or fix forward during an outage?",
         "relevant_doc_ids": [truth["incident-runbook"]]},
        {"question": "How long are client transaction records retained?",
         "relevant_doc_ids": [truth["data-retention"]]},
        {"question": "When are deletion requests actioned?",
         "relevant_doc_ids": [truth["data-retention"]]},
        {"question": "What must be recorded before promoting a model?",
         "relevant_doc_ids": [truth["model-deployment"]]},
        {"question": "Does drift automatically trigger retraining?",
         "relevant_doc_ids": [truth["model-deployment"]]},
    ]
    r = client.post("/evaluate", json={"cases": cases, "k": 3})
    out = r.json()
    print(json.dumps(out["summary"], indent=2))
    print()
    for c in out["per_case"]:
        mark = "hit " if c["hit"] else "MISS"
        print(
            f"  [{mark}] rr={c['reciprocal_rank']:.2f}  "
            f"faith={c['faithfulness']}  {c['question'][:48]}"
        )

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
