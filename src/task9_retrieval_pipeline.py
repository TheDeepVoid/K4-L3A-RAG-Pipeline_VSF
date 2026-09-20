"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc PageIndex SearchResult."""
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    normalized_query = query.strip()
    candidate_count = top_k * 2
    dense = semantic_search(normalized_query, top_k=candidate_count)
    sparse = lexical_search(normalized_query, top_k=candidate_count)

    if use_reranking:
        results = rerank_rrf([dense, sparse], top_k=top_k)
    else:
        results = dense[:top_k]

    best_dense_score = max(
        (float(item.get("score", 0.0)) for item in dense),
        default=0.0,
    )
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(normalized_query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception:
            # PageIndex là dịch vụ ngoài; kết quả retrieval hiện có vẫn dùng được.
            pass

    return results[:top_k]


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)
