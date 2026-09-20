import streamlit as st
from dotenv import load_dotenv
from src.task10_generation import generate_with_citation

load_dotenv()

st.set_page_config(
    page_title="Hỏi đáp Luật & Thuế",
    page_icon="⚖️",
    layout="wide",
)

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
        if "sources" in message and message["sources"]:
            with st.expander("📚 Nguồn trích dẫn"):
                for src in message["sources"]:
                    st.caption(f"**[{src.get('retrieval_method', 'N/A').upper()}]** Nguồn: `{src['metadata'].get('source', 'Unknown')}` (Score: {src.get('score', 0):.4f})")
                    st.text(src['content'][:200] + "...")

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
            except Exception as e:
                answer = f"Lỗi hệ thống: {str(e)}"
                sources = []
        
        st.markdown(answer)
        
        if sources:
            with st.expander("📚 Nguồn trích dẫn"):
                for src in sources:
                    st.caption(f"**[{src.get('retrieval_method', 'N/A').upper()}]** Nguồn: `{src['metadata'].get('source', 'Unknown')}` (Score: {src.get('score', 0):.4f})")
                    st.text(src['content'][:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
