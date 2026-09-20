"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.

Ghi chú SDK (pageindex 0.2.8):
    - submit_document chỉ nhận file PDF -> markdown được convert tạm sang PDF
      bằng fpdf2 (font DejaVu hỗ trợ tiếng Việt).
    - SDK không set timeout trong requests nội bộ -> bọc thêm timeout.
    - Luôn dùng PAGEINDEX_API_KEY từ .env, không hard-code.
"""

import concurrent.futures as futures
import json
import os
import re
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_documents.json"

TIMEOUT_SECONDS = 30
RETRIEVAL_ATTEMPTS = 6
RETRIEVAL_POLL_SECONDS = 2

# Font TTF hỗ trợ tiếng Việt cho fpdf2 (kiểm tra fc-list).
FONT_REGULAR = Path("/usr/share/fonts/TTF/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf")

_STATUS_DONE = {"completed", "complete", "ready", "done", "succeeded", "success", "finished"}
_STATUS_PENDING = {"queued", "queuing", "processing", "running"}


def _client():
    """Khởi tạo PageIndexClient (lazy import để test không cần SDK)."""
    from pageindex import PageIndexClient

    return PageIndexClient(PAGEINDEX_API_KEY)


def _call_with_timeout(fn, *args, timeout=TIMEOUT_SECONDS, **kwargs):
    """Chạy SDK call trong thread để áp timeout; lỗi/timeout không crash pipeline."""
    pool = futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except futures.TimeoutError as error:
        future.cancel()
        raise RuntimeError(f"PageIndex call hết thời gian chờ ({timeout}s)") from error
    finally:
        pool.shutdown(wait=False)


# ---------- Cache document IDs ----------

def _load_cache() -> dict[str, str]:
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------- Markdown -> PDF tạm ----------

def _markdown_to_plain_text(markdown: str) -> str:
    """Bỏ cú pháp markdown, giữ lại nội dung văn bản thuần."""
    text = markdown
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # image
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)      # link
    text = re.sub(r"!\{[^}]*\}", "", text)                    # markitdown block ref
    text = re.sub(r"<[^>]+>", "", text)                       # inline HTML
    text = re.sub(r"[#>*_`~|]", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    """Convert markdown sang PDF tạm để upload lên PageIndex."""
    regular = FONT_REGULAR if FONT_REGULAR.is_file() else None
    bold = FONT_BOLD if FONT_BOLD.is_file() else None
    if regular is None or bold is None:
        raise RuntimeError(
            "Không tìm thấy font Unicode (DejaVuSans) để tạo PDF. "
            f"Thiếu: {FONT_REGULAR if regular is None else FONT_BOLD}"
        )

    from fpdf import FPDF

    plain = _markdown_to_plain_text(md_path.read_text(encoding="utf-8"))
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("vn", "", str(regular))
    pdf.add_font("vn", "B", str(bold))
    pdf.add_page()
    pdf.set_font("vn", "", 10)
    for line in plain.splitlines():
        pdf.multi_cell(0, 5, line)
    pdf.output(str(pdf_path))


# ---------- Upload ----------

def _upload_one(md_path: Path) -> str:
    """Upload một file markdown, trả về doc_id từ SDK."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được cấu hình trong .env")

    with tempfile.TemporaryDirectory(prefix="pageindex_") as tmp:
        pdf_path = Path(tmp) / f"{md_path.stem}.pdf"
        _markdown_to_pdf(md_path, pdf_path)
        response = _call_with_timeout(_client().submit_document, str(pdf_path))

    doc_id = response.get("doc_id") if isinstance(response, dict) else None
    if not doc_id:
        raise RuntimeError(f"SDK trả về thiếu doc_id: {response!r}")
    return str(doc_id)


def upload_documents() -> None:
    """Upload tài liệu và lưu mapping source -> document ID."""
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY chưa được cấu hình trong .env — bỏ qua upload.")
        return
    if not STANDARDIZED_DIR.is_dir():
        print(f"Không tìm thấy thư mục: {STANDARDIZED_DIR}")
        return

    cache = _load_cache()
    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    for md_path in md_files:
        key = md_path.relative_to(STANDARDIZED_DIR).as_posix()
        if cache.get(key):
            print(f"Already uploaded: {key} -> {cache[key]}")
            continue
        try:
            doc_id = _upload_one(md_path)
        except Exception as error:
            print(f"Upload failed: {key} — {error}")
            continue
        cache[key] = doc_id
        _save_cache(cache)
        print(f"Uploaded: {key} -> {doc_id}")

    print(f"Done: {len(cache)} document(s) cached in {CACHE_FILE}")


# ---------- Retrieval ----------

def _get_status(data) -> str:
    """Đoán trường status từ các shape JSON khác nhau của API."""
    for wrapper in (data, data.get("result"), data.get("data"), data.get("response")):
        if isinstance(wrapper, dict):
            for key in ("status", "state", "processing_status"):
                value = wrapper.get(key)
                if isinstance(value, str):
                    return value.lower()
    return ""


def _extract_nodes(data) -> list[dict]:
    """Thu thập mọi dict có text/content trong response (bao gồm tree nodes)."""
    nodes: list[dict] = []

    def collect(value):
        if isinstance(value, dict):
            text = value.get("text") or value.get("content")
            if isinstance(text, str) and text.strip():
                nodes.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(data)

    # Dedupe giữ lại node có score nếu trùng text/id.
    unique: dict[tuple, dict] = {}
    for node in nodes:
        node_id = node.get("node_id") or node.get("id") or id(node)
        text = (node.get("text") or node.get("content") or "")[:200]
        key = (str(node_id), text)
        if key in unique:
            if _node_score(node) is not None and _node_score(unique[key]) is None:
                unique[key] = node
        else:
            unique[key] = node
    return list(unique.values())


def _node_score(node: dict) -> float | None:
    for key in ("score", "similarity", "relevance", "confidence"):
        value = node.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _poll_retrieval(client, retrieval_id: str) -> dict:
    """Chờ retrieval hoàn tất rồi trả về response cuối."""
    data: dict = {}
    for _ in range(RETRIEVAL_ATTEMPTS):
        data = _call_with_timeout(client.get_retrieval, retrieval_id)
        if _get_status(data) in _STATUS_DONE or _extract_nodes(data):
            return data
        time.sleep(RETRIEVAL_POLL_SECONDS)
    return data


def _query_document(client, doc_id: str, query: str) -> list[dict]:
    """Query một document, trả về danh sách retrieved nodes."""
    response = _call_with_timeout(client.submit_query, doc_id, query)
    retrieval_id = response.get("retrieval_id") if isinstance(response, dict) else None
    if not retrieval_id:
        raise RuntimeError(f"SDK trả về thiếu retrieval_id: {response!r}")
    retrieval = _poll_retrieval(client, retrieval_id)
    return _extract_nodes(retrieval)


def _to_search_result(key: str, doc_id: str, node: dict, index: int) -> dict:
    parts = Path(key).parts
    content = (node.get("text") or node.get("content") or "").strip()
    node_id = str(node.get("node_id") or node.get("id") or f"node-{index}")
    try:
        page = int(node.get("page_index", index))
    except (TypeError, ValueError):
        page = index
    return {
        "id": f"{key}::{node_id}",
        "content": content,
        "score": 0.0,
        "metadata": {
            "source": Path(key).name,
            "title": Path(key).stem,
            "doc_type": "legal" if "legal" in parts else "news",
            "url": None,
            "chunk_index": max(page, 0),
        },
        "retrieval_method": "pageindex",
    }


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if not PAGEINDEX_API_KEY:
        return []

    cache = _load_cache()
    if not cache:
        try:
            upload_documents()
        except Exception as error:
            print(f"PageIndex upload failed: {error}")
        cache = _load_cache()
    if not cache:
        return []

    client = _client()
    collected: list[tuple[float, str, str, dict, int]] = []
    global_index = 0
    for key, doc_id in cache.items():
        try:
            nodes = _query_document(client, doc_id, query)
        except Exception as error:
            print(f"PageIndex doc {key} fail: {error}")
            continue
        for node in nodes:
            score = _node_score(node)
            if score is None:
                score = 1.0 - global_index * 0.001
            collected.append((score, key, doc_id, node, global_index))
            global_index += 1

    collected.sort(key=lambda item: item[0], reverse=True)
    return [
        _to_search_result(key, doc_id, node, index)
        for score, key, doc_id, node, index in collected[:top_k]
    ]


if __name__ == "__main__":
    upload_documents()
    print("Chạy xong upload. Dùng pageindex_search(query) để truy vấn.")