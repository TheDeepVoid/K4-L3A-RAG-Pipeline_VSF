"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Điền tối thiểu 5 URL công khai. Chạy lại script sẽ ghi đè file JSON.
ARTICLE_URLS = [ 
    "https://vnexpress.net/dan-ban-hang-online-lo-bi-truy-thu-thue-4755087.html", 
    "https://cafef.vn/cu-soc-thue-voi-tiep-thi-lien-ket-nhan-ve-tui-3-ty-dong-nhung-bi-truy-thue-gan-700-trieu-dong-vi-ly-do-sau-day-188260509162820923.chn",
    "https://thuehaiquan.tapchikinhtetaichinh.vn/chinh-thuc-xoa-bo-thue-khoan-tu-2026-giai-phap-giup-ho-kinh-doanh-ke-khai-thue-dung-va-ben-vung-150688.html", 
    "https://vietnamnet.vn/ap-thue-20-lai-ban-bat-dong-san-can-lam-ro-can-cu-nao-de-tinh-tien-lai-2425198.html", 
    "https://mva.vn/bo-thue-khoan-ho-kinh-doanh-tu-nam-2026/" 
    # TODO: Thêm ít nhất 5 public URL.
]


async def crawl_article(url: str) -> dict:
    """Crawl một bài viết và trả về dict theo contract của Task 2."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    metadata = result.metadata or {}
    title = str(metadata.get("title") or "").strip() or url
    content = (result.markdown or "").strip()
    if not content:
        raise RuntimeError(f"Không trích xuất được markdown từ {url}")

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "content_markdown": content,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print(
            "ARTICLE_URLS đang trống. Thêm ít nhất 5 URL công khai "
            "vào ARTICLE_URLS rồi chạy lại."
        )
    asyncio.run(crawl_all())