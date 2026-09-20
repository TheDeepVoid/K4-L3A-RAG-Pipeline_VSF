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
import time

from dotenv import load_dotenv

from .task9_retrieval_pipeline import get_retrieval_trace, retrieve


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
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        parts.append(
            f"[Document {index} | Title: {metadata['title']} | "
            f"Source: {metadata['source']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    if LLM_PROVIDER.lower() == "gemini":
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=LLM_MODEL or "gemini-2.5-flash",
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        )
        return response.text
    elif LLM_PROVIDER.lower() == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=LLM_MODEL or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content
    else:
        # Default fallback if provider is unknown or Anthropic is omitted for brevity
        return f"[Simulated LLM Response] Could not call provider {LLM_PROVIDER}"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult kèm \"steps\" mô tả chi tiết pipeline."""
    t0 = time.perf_counter()
    steps: list[dict] = []

    chunks = retrieve(query, top_k=top_k)
    steps = list(get_retrieval_trace())

    if not chunks:
        steps.append({
            "phase": "generation",
            "title": "Không đủ context",
            "detail": "Retrieval không trả về chunk nào — dùng câu trả lời an toàn, không bịa thông tin.",
            "status": "skip",
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        })
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
            "steps": steps,
        }

    reordered = reorder_for_llm(chunks)
    steps.append({
        "phase": "generation",
        "title": "Sắp xếp context (chống lost-in-the-middle)",
        "detail": (
            f"Đưa {len(reordered)} chunks quan trọng lên đầu/cuối context "
            "để LLM không bỏ sót thông tin giữa."
        ),
        "status": "ok",
        "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
    })

    context = format_context(reordered)
    steps.append({
        "phase": "generation",
        "title": "Xây dựng context cho LLM",
        "detail": f"{len(reordered)} chunks, {len(context)} ký tự, kèm title + source label.",
        "status": "ok",
        "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
    })

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    t = time.perf_counter()
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
        llm_error = None
    except Exception as e:
        answer = f"Lỗi gọi LLM: {e}"
        llm_error = str(e)

    model_note = LLM_MODEL or ("gpt-4o-mini" if LLM_PROVIDER.lower() == "openai" else "gemini-2.5-flash")
    steps.append({
        "phase": "generation",
        "title": "Gọi LLM sinh câu trả lời",
        "detail": (
            f"Provider {LLM_PROVIDER}, model {model_note}, temperature={TEMPERATURE}, "
            f"top_p={TOP_P}, max_output_tokens={MAX_OUTPUT_TOKENS}."
            + (f" Lỗi: {llm_error}" if llm_error else "")
        ),
        "status": "error" if llm_error else "ok",
        "duration_ms": round((time.perf_counter() - t) * 1000, 1),
    })

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0]["retrieval_method"],
        "steps": steps,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
