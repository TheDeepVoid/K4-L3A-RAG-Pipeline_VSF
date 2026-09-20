"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Điền tối thiểu 3 cặp "tên file (không dấu)" -> "URL công khai".
# Chạy lại script sẽ bỏ qua file đã tải thành công (idempotent).
SOURCES: dict[str, str] = {
    # Ví dụ: "policy-a.pdf": "https://example.edu/policy-a.pdf",
    "luat-108-2025-qh15_1612185241.docx": "https://static3.luatvietnam.vn/uploaded/others/2025/12/16/luat-108-2025-qh15_1612185241.docx",
    "luat-67-2025-qh15_1809150248.docx": "https://static3.luatvietnam.vn/uploaded/others/2025/09/18/luat-67-2025-qh15_1809150248.docx",
    "d3cad3c4d68b4a75b95fcd5e8562fa5e.docx": "https://gatewayduthaoonline.quochoi.vn/uploadFiles/host/local/2025/12/10/10/d3cad3c4d68b4a75b95fcd5e8562fa5e.docx",
}

TIMEOUT_SECONDS = 30
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def _looks_like_html(content: bytes) -> bool:
    """Nhận diện trang HTML trả về thay vì file PDF/DOCX."""
    head = content[:512].lstrip().lower()
    return head.startswith(b"<html") or head.startswith(b"<!doctype")


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    if not SOURCES:
        print(
            "SOURCES đang trống. Thêm ít nhất 3 cặp filename -> URL "
            "vào SOURCES rồi chạy lại."
        )
        return

    headers = {"User-Agent": USER_AGENT}
    for filename, url in SOURCES.items():
        target = DATA_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"Skip existing: {target}")
            continue
        try:
            response = requests.get(url, timeout=TIMEOUT_SECONDS, headers=headers)
            response.raise_for_status()
            if not response.content:
                raise ValueError("response body rỗng")
            if _looks_like_html(response.content):
                raise ValueError("URL trả về HTML thay vì file PDF/DOCX")
            target.write_bytes(response.content)
            print(f"Saved: {target} ({len(response.content)} bytes)")
        except Exception as error:
            print(f"Failed: {filename} — {error}")


if __name__ == "__main__":
    setup_directory()
    download_documents()