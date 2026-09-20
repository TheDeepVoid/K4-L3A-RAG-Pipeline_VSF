"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.

Trace: mỗi lần retrieve() ghi chi tiết từng bước vào module-level TRACE để UI
(app.py) hiển thị quá trình tìm kiếm trước khi hiện kết quả. Việc ghi trace
không thay đổi giá trị trả về của retrieve().
"""

import time

from .task4_chunking_indexing import EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_PROVIDER
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5

# Trace của lần retrieve() gần nhất (chỉ để hiển thị, không phải dữ liệu trả về).
TRACE: list[dict] = []


def _record(step: dict) -> None:
    """Append một bước vào TRACE — không được raise dù step thiếu dữ liệu."""
    try:
        TRACE.append(step)
    except Exception:
        pass


def _step_stats(results: list[dict]) -> dict:
    """Rút gọn stats an toàn từ SearchResult (chịu được dữ liệu thiếu key)."""
    try:
        scores = [float(item.get("score", 0.0)) for item in results]
        return {
            "count": len(scores),
            "best_score": round(max(scores, default=0.0), 6),
            "top_id": results[0].get("id") if results else None,
            "method": results[0].get("retrieval_method") if results else None,
        }
    except Exception:
        return {"count": 0, "best_score": 0.0, "top_id": None, "method": None}


def get_retrieval_trace() -> list[dict]:
    """Trả về các bước của lần retrieve() gần nhất (bản copy)."""
    return list(TRACE)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc PageIndex SearchResult + ghi trace từng bước."""
    TRACE.clear()
    trace_start = time.perf_counter()

    def elapsed() -> float:
        return round((time.perf_counter() - trace_start) * 1000, 1)

    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        _record({
            "phase": "retrieval",
            "title": "Từ chối truy vấn",
            "detail": "Query rỗng hoặc top_k không hợp lệ — không thực hiện tìm kiếm.",
            "status": "skip",
            "duration_ms": elapsed(),
        })
        return []

    normalized_query = query.strip()
    candidate_count = top_k * 2

    _record({
        "phase": "retrieval",
        "title": "Chuẩn bị câu hỏi",
        "detail": (
            f"Query: “{normalized_query[:120]}” | top_k={top_k} | candidates={candidate_count} "
            f"| embedding {EMBEDDING_PROVIDER}/{EMBEDDING_MODEL} ({EMBEDDING_DIM}D)"
        ),
        "status": "ok",
        "duration_ms": elapsed(),
    })

    t = time.perf_counter()
    dense = semantic_search(normalized_query, top_k=candidate_count)
    stats = _step_stats(dense)
    _record({
        "phase": "retrieval",
        "title": "Semantic search (dense)",
        "detail": (
            f"Embed query {EMBEDDING_DIM}D → tra Chroma (cosine). "
            f"Tìm {stats['count']} chunks, điểm cao nhất {stats['best_score']} "
            f"({stats['top_id']})"
        ),
        "status": "ok",
        "duration_ms": round((time.perf_counter() - t) * 1000, 1),
    })

    t = time.perf_counter()
    sparse = lexical_search(normalized_query, top_k=candidate_count)
    stats = _step_stats(sparse)
    _record({
        "phase": "retrieval",
        "title": "Lexical search (BM25)",
        "detail": (
            f"Khớp từ khóa trên {stats['count']} chunks, điểm cao nhất {stats['best_score']} "
            f"({stats['top_id']})"
        ),
        "status": "ok" if stats["count"] else "warn",
        "duration_ms": round((time.perf_counter() - t) * 1000, 1),
    })

    t = time.perf_counter()
    if use_reranking:
        results = rerank_rrf([dense, sparse], top_k=top_k)
        stats = _step_stats(results)
        _record({
            "phase": "retrieval",
            "title": "Fusion bằng RRF",
            "detail": (
                f"Gộp {len(dense)} dense + {len(sparse)} BM25 theo 1/(60+rank), giữ top {top_k}. "
                f"Điểm RRF cao nhất {stats['best_score']} ({stats['top_id']})"
            ),
            "status": "ok",
            "duration_ms": round((time.perf_counter() - t) * 1000, 1),
        })
    else:
        results = dense[:top_k]
        _record({
            "phase": "retrieval",
            "title": "Bỏ qua rerank",
            "detail": "use_reranking=False — giữ nguyên top_k từ dense search.",
            "status": "skip",
            "duration_ms": round((time.perf_counter() - t) * 1000, 1),
        })

    best_dense_score = max(
        (float(item.get("score", 0.0)) for item in dense),
        default=0.0,
    )
    fallback_triggered = best_dense_score < score_threshold
    _record({
        "phase": "retrieval",
        "title": "Kiểm tra fallback (PageIndex)",
        "detail": (
            f"Best dense score = {round(best_dense_score, 4)} (threshold {score_threshold}) → "
            + ("KÍCH HOẠT fallback." if fallback_triggered else "Đủ tin cậy, giữ kết quả RRF.")
        ),
        "status": "warn" if fallback_triggered else "ok",
        "duration_ms": elapsed(),
    })

    if fallback_triggered:
        t = time.perf_counter()
        try:
            fallback = pageindex_search(normalized_query, top_k=top_k)
            if fallback:
                _record({
                    "phase": "retrieval",
                    "title": "Fallback PageIndex",
                    "detail": f"Trả {len(fallback)} kết quả PageIndex thay cho RRF.",
                    "status": "ok",
                    "duration_ms": round((time.perf_counter() - t) * 1000, 1),
                })
                return fallback[:top_k]
            _record({
                "phase": "retrieval",
                "title": "Fallback PageIndex",
                "detail": "PageIndex không có kết quả — giữ kết quả RRF.",
                "status": "skip",
                "duration_ms": round((time.perf_counter() - t) * 1000, 1),
            })
        except Exception:
            _record({
                "phase": "retrieval",
                "title": "Fallback PageIndex",
                "detail": "PageIndex lỗi (dịch vụ ngoài) — giữ kết quả RRF thay vì crash.",
                "status": "error",
                "duration_ms": round((time.perf_counter() - t) * 1000, 1),
            })

    return results[:top_k]


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)