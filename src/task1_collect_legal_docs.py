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


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    import urllib3
    import requests
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    sources = {
        "luat_108_2025_qh15.docx": "https://static3.luatvietnam.vn/uploaded/others/2025/12/16/luat-108-2025-qh15_1612185241.docx",
        "duthao_luat_2025.docx": "https://gatewayduthaoonline.quochoi.vn/uploadFiles/host/local/2025/12/10/10/d3cad3c4d68b4a75b95fcd5e8562fa5e.docx",
        "luat_67_2025_qh15.docx": "https://static3.luatvietnam.vn/uploaded/others/2025/09/18/luat-67-2025-qh15_1809150248.docx",
    }
    for filename, url in sources.items():
        print(f"Downloading {filename}...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, timeout=30, verify=False, headers=headers)
        response.raise_for_status()
        (DATA_DIR / filename).write_bytes(response.content)
        print(f"Saved {filename}")


if __name__ == "__main__":
    setup_directory()
    download_documents()
