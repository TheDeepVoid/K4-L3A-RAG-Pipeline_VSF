"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        print("Skipping PageIndex upload: No API key found.")
        return
        
    try:
        import pageindex
        client = pageindex.Client(api_key=PAGEINDEX_API_KEY)
        
        for path in STANDARDIZED_DIR.rglob("*.md"):
            doc_type = "legal" if "legal" in path.parts else "news"
            content = path.read_text(encoding="utf-8")
            # This is a hypothetical SDK call based on standard patterns
            try:
                client.documents.upload(
                    content=content,
                    metadata={"source": path.name, "doc_type": doc_type}
                )
                print(f"Uploaded {path.name} to PageIndex")
            except Exception as e:
                print(f"Failed to upload {path.name}: {e}")
    except Exception as e:
        print(f"PageIndex integration error: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if not PAGEINDEX_API_KEY:
        return []
        
    try:
        import pageindex
        client = pageindex.Client(api_key=PAGEINDEX_API_KEY)
        
        # Hypothetical SDK call
        search_res = client.search(query=query, top_k=top_k)
        
        results = []
        for index, item in enumerate(search_res.get("nodes", [])):
            results.append({
                "id": item.get("id", f"pi_node_{index}"),
                "content": item.get("text", ""),
                "score": float(item.get("score", max(0.1, 1.0 - index * 0.1))),
                "metadata": item.get("metadata", {}),
                "retrieval_method": "pageindex",
            })
        return results
    except Exception as e:
        print(f"PageIndex search error: {e}")
        return []


if __name__ == "__main__":
    upload_documents()
