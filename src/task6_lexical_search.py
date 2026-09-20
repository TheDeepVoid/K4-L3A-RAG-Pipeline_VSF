"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.

Corpus: nạp lazy từ Chroma collection của Task 4 khi CORPUS rỗng, hoặc được seed
trực tiếp bởi task4.run_pipeline() để khỏi đọc lại DB. Nếu DB rỗng/lỗi thì
lexical_search trả [] thay vì crash.
"""


CORPUS: list[dict] = []

_bm25_index = None
_bm25_corpus_size = 0


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Plus
    tokenized = [item["content"].lower().split() for item in corpus]
    return BM25Plus(tokenized)


def _ensure_corpus() -> None:
    """Nạp corpus từ Chroma collection của Task 4 nếu CORPUS còn rỗng."""
    global CORPUS
    if CORPUS:
        return
    try:
        from .task4_chunking_indexing import get_collection

        collection = get_collection()
        data = collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        CORPUS = [
            {"id": item_id, "content": content, "metadata": metadata or {}}
            for item_id, content, metadata in zip(ids, documents, metadatas)
        ]
    except Exception:
        # Chroma chưa có dữ liệu hoặc lỗi — giữ CORPUS rỗng, lexical_search trả [].
        CORPUS = []


def _get_bm25():
    """Trả BM25 index, rebuild nếu corpus đổi kích thước (hoặc vừa được seed)."""
    global _bm25_index, _bm25_corpus_size
    if not CORPUS:
        _ensure_corpus()
    if not CORPUS:
        return None
    if _bm25_index is None or _bm25_corpus_size != len(CORPUS):
        _bm25_index = build_bm25_index(CORPUS)
        _bm25_corpus_size = len(CORPUS)
    return _bm25_index


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    import numpy as np

    bm25 = _get_bm25()
    if bm25 is None:
        return []

    scores = bm25.get_scores(query.lower().split())
    indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for index in indices:
        if scores[index] <= 0:
            continue
        item = CORPUS[index]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)