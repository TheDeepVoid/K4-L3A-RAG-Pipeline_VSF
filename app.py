import streamlit as st
from dotenv import load_dotenv
from src.task10_generation import generate_with_citation

load_dotenv()

st.set_page_config(
    page_title="Hỏi đáp Luật & Thuế",
    page_icon="⚖️",
    layout="wide",
)

STEP_ICONS = {
    "ok": "✅",
    "warn": "⚠️",
    "error": "❌",
    "skip": "⏭️",
}


def render_steps(steps):
    """Hiển thị chi tiết từng bước pipeline tra cứu của LLM."""
    if not steps:
        return
    title = f"🔍 Các bước tra cứu cơ sở dữ liệu ({len(steps)} bước)"
    with st.expander(title, expanded=True):
        for index, step in enumerate(steps, 1):
            icon = STEP_ICONS.get(step.get("status", "ok"), "ℹ️")
            duration = step.get("duration_ms")
            duration_note = f" — {duration:.0f}ms" if isinstance(duration, (int, float)) else ""
            st.markdown(f"{icon} **Bước {index}. {step.get('title', '')}**{duration_note}")
            if step.get("detail"):
                st.caption(step["detail"])


def render_sources(sources):
    """Hiển thị các chunk trích dẫn kèm phương pháp retrieval."""
    if not sources:
        return
    with st.expander("📚 Nguồn trích dẫn"):
        for src in sources:
            st.caption(f"**[{src.get('retrieval_method', 'N/A').upper()}]** Nguồn: `{src['metadata'].get('source', 'Unknown')}` (Score: {src.get('score', 0):.4f})")
            st.text(src['content'][:200] + "...")


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("⚖️ Hệ thống RAG")
    st.caption("Tra cứu văn bản Pháp luật và Tin tức về Thuế")
    top_k = st.slider("Số lượng tài liệu trích xuất (Chunks)", 3, 10, 5)

st.title("Trợ lý Pháp lý & Thuế 🤖")
st.caption("Hãy đặt câu hỏi về Luật, Dự thảo Luật hoặc Thuế Doanh nghiệp. Hệ thống sẽ trích xuất từ dữ liệu nội bộ để trả lời.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_steps(message.get("steps", []))
        render_sources(message.get("sources", []))

query = st.chat_input("Ví dụ: Mức phạt chậm nộp thuế là bao nhiêu?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu cơ sở dữ liệu..."):
            try:
                result = generate_with_citation(query, top_k=top_k)
                answer = result["answer"]
                sources = result.get("sources", [])
                steps = result.get("steps", [])
            except Exception as e:
                answer = f"Lỗi hệ thống: {str(e)}"
                sources = []
                steps = []

        render_steps(steps)

        st.markdown(answer)

        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "steps": steps}
    )