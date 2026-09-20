"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Browser: dùng Brave có sẵn trên máy qua CDP thay vì tải playwright chromium.
    - Playwright không có channel "brave", nhưng crawl4ai kết nối được browser
      bất kỳ (chromium-based) qua CDP: BrowserConfig(cdp_url=...).
    - Script tự mở Brave với --remote-debugging-port rồi crawl qua CDP.
    - Sửa BRAVE_EXECUTABLE nếu máy đặt Brave ở đường dẫn khác.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Điền tối thiểu 5 URL công khai. Chạy lại script sẽ ghi đè file JSON.
ARTICLE_URLS = [
    "https://vnexpress.net/dan-ban-hang-online-lo-bi-truy-thu-thue-4755087.html",
    "https://cafef.vn/cu-soc-thue-voi-tiep-thi-lien-ket-nhan-ve-tui-3-ty-dong-nhung-bi-truy-thue-gan-700-trieu-dong-vi-ly-do-sau-day-188260509162820923.chn",
    "https://thuehaiquan.tapchikinhtetaichinh.vn/chinh-thuc-xoa-bo-thue-khoan-tu-2026-giai-phap-giup-ho-kinh-doanh-ke-khai-thue-dung-va-ben-vung-150688.html",
    "https://vietnamnet.vn/ap-thue-20-lai-ban-bat-dong-san-can-lam-ro-can-cu-nao-de-tinh-tien-lai-2425198.html",
    "https://mva.vn/bo-thue-khoan-ho-kinh-doanh-tu-nam-2026/"
]

# --- Brave qua CDP -------------------------------------------------------
BRAVE_EXECUTABLE = "/usr/bin/brave"
CDP_HOST = "127.0.0.1"
CDP_PORT = 9222
CDP_URL = f"http://{CDP_HOST}:{CDP_PORT}"
CDP_READY_TIMEOUT = 20

_BRAVE_PROC: subprocess.Popen | None = None
_BRAVE_PROFILE: Path | None = None


def _cdp_ready(timeout: float = 2.0) -> bool:
    try:
        response = requests.get(f"{CDP_URL}/json/version", timeout=timeout)
        return response.ok
    except requests.RequestException:
        return False


def get_brave_cdp_url() -> str:
    """Đảm bảo Brave đang chạy với CDP port, trả về CDP URL để crawl."""
    global _BRAVE_PROC, _BRAVE_PROFILE

    # Đã có Brave do script này mở và còn sống -> dùng lại.
    if _BRAVE_PROC is not None and _BRAVE_PROC.poll() is None:
        return CDP_URL
    # Có browser khác (mở tay/ngoài) đang nghe port -> tái sử dụng.
    if _cdp_ready():
        return CDP_URL

    if not Path(BRAVE_EXECUTABLE).is_file():
        raise RuntimeError(
            f"Không tìm thấy {BRAVE_EXECUTABLE}. Sửa BRAVE_EXECUTABLE "
            "hoặc dùng AsyncWebCrawler() mặc định (cần playwright chromium)."
        )

    _BRAVE_PROFILE = Path(tempfile.mkdtemp(prefix="brave-cdp-"))
    command = [
        BRAVE_EXECUTABLE,
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={_BRAVE_PROFILE}",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-gpu",
        "--headless=new",
    ]
    try:
        if sys.platform == "win32":
            _BRAVE_PROC = subprocess.Popen(command)
        else:
            _BRAVE_PROC = subprocess.Popen(
                command, start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    except OSError as error:
        raise RuntimeError(f"Không mở được Brave: {error}") from error

    deadline = time.monotonic() + CDP_READY_TIMEOUT
    while time.monotonic() < deadline:
        if _BRAVE_PROC.poll() is not None:
            break
        if _cdp_ready():
            print(f"Brave CDP ready: {CDP_URL}")
            return CDP_URL
        time.sleep(0.5)

    raise RuntimeError(
        f"Brave không sẵn sàng CDP sau {CDP_READY_TIMEOUT}s. "
        f"Exit code: {_BRAVE_PROC.poll()}"
    )


def shutdown_brave() -> None:
    """Tắt Brave mà script đã mở (kèm cả process con)."""
    global _BRAVE_PROC, _BRAVE_PROFILE
    proc, profile = _BRAVE_PROC, _BRAVE_PROFILE
    _BRAVE_PROC, _BRAVE_PROFILE = None, None
    if proc is None:
        return
    try:
        if sys.platform == "win32":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    if profile is not None:
        try:
            import shutil

            shutil.rmtree(profile, ignore_errors=True)
        except OSError:
            pass


async def crawl_article(url: str) -> dict:
    from datetime import datetime
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        
        title = "Unknown"
        if result.metadata and isinstance(result.metadata, dict):
            title = result.metadata.get("title", "Unknown")
            
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown,
        }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        get_brave_cdp_url()
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
    finally:
        shutdown_brave()


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print(
            "ARTICLE_URLS đang trống. Thêm ít nhất 5 URL công khai "
            "vào ARTICLE_URLS rồi chạy lại."
        )
    asyncio.run(crawl_all())