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
import re
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


# --- Guardrail / prompt protection -------------------------------------------
# Nhận diện câu hỏi ngoài lĩnh vực và trích xuất câu hỏi thật từ prompt dài
# (bị nhồi nhét nội dung không liên quan, e.g. lời bài hát).

DOMAIN_SIGNALS = [
    "thuế", "luật", "pháp luật", "doanh nghiệp", "quy định", "nghị định",
    "thông tư", "phạt", "kê khai", "hóa đơn", "hoá đơn", "thu nhập",
    "quyết toán", "tạm tính", "chi phí", "khấu trừ", "hộ kinh doanh",
    "đăng ký kinh doanh", "học phí", "ký túc xá", "sinh viên", "đại học",
    "thời hạn", "tờ khai", "miễn thuế", "ưu đãi", "hợp đồng", "nhà nước",
    "lương", "bảo hiểm", "đất đai", "xuất nhập khẩu", "chậm nộp",
]

QUESTION_MARKERS = [
    "?", "bao nhiêu", "thế nào", "như thế nào", "là gì", "tại sao",
    "khi nào", "ở đâu", "mức", "có được không", "được không",
    "phải không", "nếu", "nên", "có không", "gì",
]

INJECTION_MARKERS = [
    "bỏ qua", "ignore", "hướng dẫn trước", "system prompt", "system:",
    "đóng vai", "act as", "you are", "không được trả lời", "đừng trả lời",
    "quên đi", "prompt", "developer message", "bạn là",
]

REJECT_MESSAGE = (
    "Câu hỏi của bạn không liên quan đến lĩnh vực pháp luật và thuế mà hệ "
    "thống hỗ trợ, nên tôi không thể trả lời. Hãy hỏi về quy định, mức thuế, "
    "thủ tục hoặc tin tức trong bộ dữ liệu của chúng tôi."
)


def _normalize(text: str) -> str:
    """Gộp khoảng trắng thừa để dễ so sánh/scoring."""
    return " ".join(text.split())


def _split_units(text: str) -> list[str]:
    """Tách prompt thành các đơn vị câu (theo dòng hoặc dấu '?')."""
    parts = re.split(r"[\n?]+", text)
    return [part.strip() for part in parts if part.strip()]


def _score_unit(unit: str) -> float:
    """Chấm điểm mức độ 'đúng lĩnh vực + là câu hỏi' của một đơn vị văn bản."""
    low = unit.lower()
    domain_hits = sum(low.count(signal) for signal in DOMAIN_SIGNALS)
    has_question = any(marker in low for marker in QUESTION_MARKERS)
    has_injection = any(marker in low for marker in INJECTION_MARKERS)
    if has_injection:
        return -1.0
    return 2.0 * domain_hits + (1.0 if has_question else 0.0)


def guard_query(query: str) -> dict:
    """
    Guardrail: kiểm tra mức độ liên quan và trích xuất câu hỏi thật.

    Trả về dict:
        - status: "ok" | "rejected"
        - query: câu hỏi (đã trích xuất/chuẩn hóa) dùng cho retrieval
        - extracted: True nếu đã rút câu hỏi ra khỏi prompt dài
        - original: bản chuẩn hóa của input
        - reason: lý do từ chối (nếu rejected)
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "status": "rejected",
            "query": "",
            "extracted": False,
            "original": "",
            "reason": "Câu hỏi trống hoặc không hợp lệ.",
        }

    normalized = _normalize(query)
    units = _split_units(query)

    candidate = None
    if len(units) > 1:
        best_unit, best_score = max(
            ((unit, _score_unit(unit)) for unit in units),
            key=lambda pair: pair[1],
        )
        if best_score >= 2.0:
            candidate = best_unit

    chosen = candidate or normalized

    if candidate:
        return {
            "status": "ok",
            "query": candidate,
            "extracted": True,
            "original": normalized,
            "reason": "",
        }

    score = _score_unit(normalized)
    if score < 1.0:
        return {
            "status": "rejected",
            "query": normalized,
            "extracted": False,
            "original": normalized,
            "reason": (
                "Không tìm thấy dấu hiệu liên quan đến pháp luật/thuế "
                "(không có từ khóa lĩnh vực hoặc dấu hiệu câu hỏi)."
            ),
        }

    return {
        "status": "ok",
        "query": normalized,
        "extracted": False,
        "original": normalized,
        "reason": "",
    }


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult kèm \"steps\" mô tả chi tiết pipeline.

    Guardrail: câu hỏi ngoài lĩnh vực bị từ chối; prompt dài có nhồi nội dung
    không liên quan sẽ được trích xuất câu hỏi thật trước khi retrieval.
    """
    t0 = time.perf_counter()
    steps: list[dict] = []

    guarded = guard_query(query)
    guard_steps = [
        {
            "phase": "guard",
            "title": "Guardrail: kiểm tra câu hỏi",
            "detail": f"Query gốc: “{guarded['original'][:150]}”",
            "status": "ok",
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    ]

    if guarded["status"] == "rejected":
        guard_steps.append({
            "phase": "guard",
            "title": "Từ chối câu hỏi ngoài lĩnh vực",
            "detail": guarded["reason"],
            "status": "warn",
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        })
        return {
            "answer": REJECT_MESSAGE,
            "sources": [],
            "retrieval_source": "none",
            "steps": guard_steps,
        }

    if guarded["extracted"]:
        guard_steps.append({
            "phase": "guard",
            "title": "Trích xuất câu hỏi thật từ prompt",
            "detail": (
                f"Nội dung ngoài lĩnh vực đã được lọc. Câu hỏi dùng cho tra cứu: "
                f"“{guarded['query'][:150]}”"
            ),
            "status": "ok",
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    used_query = guarded["query"]
    chunks = retrieve(used_query, top_k=top_k)
    steps = guard_steps + list(get_retrieval_trace())

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

    user_message = f"Context:\n{context}\n\nQuestion: {used_query}"

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
