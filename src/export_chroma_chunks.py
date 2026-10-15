"""
dump_chunks.py — export all stored chunks from the Chroma store to JSON.

Assumes your retriever code lives in src/retriever.py and exposes load_store()
(which opens the persisted Chroma store without rebuilding).

Run from the project root:
    python dump_chunks.py

Produces chunks_dump.json — every chunk's id, text, and metadata — so you can
scan them (e.g. to hand-pick ideal_context for the faithfulness dataset).
"""

import json
from src.retriever import load_store   # reuse the retriever's store loader

# open the already-built store (no re-embedding)
store = load_store()

# pull every stored chunk. include=["documents","metadatas"] is required —
# by default Chroma returns ids + metadata but NOT the chunk text.
data = store._collection.get(include=["documents", "metadatas"])

# zip the parallel lists into one record per chunk
dump = [
    {"id": i, "text": d, "meta": m}
    for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
]

# sort by session then by id so related chunks sit together (easier to scan)
dump.sort(key=lambda c: (str(c["meta"].get("session", "")), c["id"]))

with open("goldens/chunks_dump.json", "w") as f:
    json.dump(dump, f, indent=2, ensure_ascii=False)

print(f"Dumped {len(dump)} chunks to goldens/chunks_dump.json")

# quick per-session count so you can see the spread
from collections import Counter
counts = Counter(str(c["meta"].get("session", "?")) for c in dump)
for session in sorted(counts):
    print(f"  session {session}: {counts[session]} chunks")