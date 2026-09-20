"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 2048

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp dựa trên tài liệu.
Chỉ trả lời từ context được cung cấp và không bổ sung thông tin bên ngoài.
Sau mỗi khẳng định, trích dẫn một hoặc nhiều nguồn bằng đúng token [Source: <id>]
có trong context. Nếu context không đủ evidence, chỉ trả lời câu từ chối được
cung cấp trong hướng dẫn của người dùng.
"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context mà không mutate input."""
    ordered = list(chunks)
    if len(ordered) <= 2:
        return ordered

    front = ordered[::2]
    back = ordered[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có citation ID, title, source và URL."""
    parts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        content = str(chunk.get("content", "")).strip()
        if not content:
            continue

        metadata = chunk.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        chunk_id = str(chunk.get("id") or "unknown")
        title = str(metadata.get("title") or "Unknown")
        source = str(metadata.get("source") or "Unknown")
        url = metadata.get("url")

        labels = [
            f"Source: {chunk_id}",
            f"Title: {title}",
            f"File: {source}",
        ]
        if url:
            labels.append(f"URL: {url}")
        parts.append(f"[{' | '.join(labels)}]\n{content}")

    return "\n\n---\n\n".join(parts)


def _require_setting(name: str, value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"Missing required setting: {name}")
    return value


def _call_openai(system_prompt: str, user_message: str, model: str) -> str:
    from openai import OpenAI

    api_key = _require_setting("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_message,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return str(response.output_text or "").strip()


def _call_gemini(system_prompt: str, user_message: str, model: str) -> str:
    from google import genai
    from google.genai import types

    api_key = _require_setting("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    return str(response.text or "").strip()


def _call_anthropic(system_prompt: str, user_message: str, model: str) -> str:
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = LLM_PROVIDER.strip().lower()
    model = _require_setting("LLM_MODEL", LLM_MODEL)

    dispatch = {
        "openai": _call_openai,
        "gemini": _call_gemini,
        "anthropic": _call_anthropic,
    }
    if provider not in dispatch:
        supported = ", ".join(dispatch)
        raise ValueError(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. Expected one of: {supported}"
        )

    answer = dispatch[provider](system_prompt, user_message, model)
    if not answer:
        raise RuntimeError(f"{provider} returned an empty response")
    return answer


def _safe_refusal() -> dict:
    return {
        "answer": SAFE_REFUSAL,
        "sources": [],
        "retrieval_source": "none",
    }


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Retrieve evidence, generate a cited answer và trả GenerationResult."""
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return _safe_refusal()

    try:
        chunks = retrieve(query.strip(), top_k=top_k)
        if not chunks:
            return _safe_refusal()

        context = format_context(reorder_for_llm(chunks))
        if not context:
            return _safe_refusal()

        user_message = (
            f"Context:\n{context}\n\nQuestion: {query.strip()}\n\n"
            f"Nếu evidence không đủ, trả lời chính xác: {SAFE_REFUSAL}"
        )
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception:
        return _safe_refusal()

    retrieval_source = (
        "pageindex"
        if chunks[0].get("retrieval_method") == "pageindex"
        else "hybrid"
    )
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
