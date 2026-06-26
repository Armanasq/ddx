import re
import sqlite3

import requests


class Retriever:
    def __init__(self, config):
        self.config = config

    def search(self, query, limit=5):
        # Hybrid retrieval fused by Reciprocal Rank Fusion. Ranking uses ONLY the two
        # engines' own rank positions — BM25 (lexical) and dense vectors (semantic).
        # No hand-coded terms, keywords, disease names, or population patterns are used:
        # relevance emerges from the retrievers' agreement, and final relevance judgement
        # is left to the LLM verification and synthesis steps.
        sql_hits = self.sql_search(query, limit=limit * 3)
        qdrant_hits = self.qdrant_search(query, limit=limit * 3)
        return reciprocal_rank_fusion([sql_hits, qdrant_hits], limit=limit)

    def sql_search(self, query, limit=10):
        con = sqlite3.connect(self.config.knowledge_db)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            select c.chunk_id, c.doc_id, d.title, s.section_path, c.text, bm25(chunks_fts) score
            from chunks_fts
            join chunks c on c.chunk_id = chunks_fts.chunk_id
            join documents d on d.doc_id = c.doc_id
            join sections s on s.section_id = c.section_id
            where chunks_fts match ?
            order by score
            limit ?
            """,
            (fts_query(query), limit),
        ).fetchall()
        con.close()
        return [pack(row) for row in rows]

    def qdrant_search(self, query, limit=10):
        vector = self.embed(query)
        url = f"{self.config.qdrant_url.rstrip('/')}/collections/{self.config.qdrant_collection}/points/query"
        response = requests.post(
            url,
            json={
                "query": vector,
                "using": "dense",
                "limit": limit,
                "with_payload": ["chunk_id"],
                "with_vector": False,
            },
            timeout=30,
        )
        response.raise_for_status()
        points = response.json()["result"]["points"]
        ids = [p["payload"]["chunk_id"] for p in points if p.get("payload", {}).get("chunk_id")]
        return self.fetch_chunks(ids)

    def embed(self, text):
        response = requests.post(
            self.config.embedding_uri,
            headers={"api-key": self.config.embedding_auth, "content-type": "application/json"},
            json={"input": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def fetch_chunks(self, ids):
        if not ids:
            return []
        con = sqlite3.connect(self.config.knowledge_db)
        con.row_factory = sqlite3.Row
        marks = ",".join("?" for _ in ids)
        rows = con.execute(
            f"""
            select c.chunk_id, c.doc_id, d.title, s.section_path, c.text, 0 score
            from chunks c
            join documents d on d.doc_id = c.doc_id
            join sections s on s.section_id = c.section_id
            where c.chunk_id in ({marks})
            """,
            ids,
        ).fetchall()
        con.close()
        by_id = {row["chunk_id"]: pack(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]


def fts_query(text):
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", text.lower())
    return " OR ".join(f'"{token}"' for token in tokens[:12])


def pack(row):
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "title": row["title"],
        "section_path": row["section_path"],
        "text": row["text"][:900],
    }


def reciprocal_rank_fusion(ranked_lists, limit=5, k=60):
    """Fuse multiple ranked result lists by Reciprocal Rank Fusion.

    score(d) = sum over lists of 1 / (k + rank_in_list(d))

    Uses only rank positions — no lexical scores, no term/keyword/pattern matching.
    An item ranked highly by both retrievers accumulates the most score; an item that
    only one retriever surfaced (typical recall noise) is pushed down.
    """
    scores = {}
    items = {}
    for hits in ranked_lists:
        for rank, item in enumerate(hits):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            items[cid] = item
    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
    out = []
    for cid in ordered[:limit]:
        item = items[cid]
        item["fit"] = scores[cid]
        out.append(item)
    return out
