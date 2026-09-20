"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Embedding: dùng OpenAI API (key trong .env), không tải model local.
    - EMBEDDING_PROVIDER=openai (mặc định): gọi embeddings API của OpenAI.
    - EMBEDDING_PROVIDER=sentence_transformers: chạy model local (fallback).

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Provider embedding chọn trong .env: openai (mặc định) | sentence_transformers.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower().strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))

COLLECTION_NAME = "rag_documents"

_openai_client = None
_st_model = None


def _get_openai_client():
    """Singleton cho OpenAI client (đọc OPENAI_API_KEY từ .env)."""
    global _openai_client
    if _openai_client is None:
        if not os.getenv("OPENAI_API_KEY", ""):
            raise RuntimeError(
                "OPENAI_API_KEY chưa được cấu hình. Thêm key vào .env "
                "(xem .env.example) rồi chạy lại."
            )
        from openai import OpenAI

        _openai_client = OpenAI()
    return _openai_client


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """Embed bằng OpenAI embeddings API, gọi theo batch."""
    client = _get_openai_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend([item.embedding for item in response.data])
    return vectors


def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    """Fallback: model local qua sentence-transformers (load một lần)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(EMBEDDING_MODEL)
    return _st_model.encode(texts).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed chủ đề theo EMBEDDING_PROVIDER trong .env (OpenAI API hoặc model local)."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts)
    if EMBEDDING_PROVIDER in {"sentence_transformers", "sentence-transformers", "local"}:
        return _embed_sentence_transformers(texts)
    raise ValueError(
        f"EMBEDDING_PROVIDER không hỗ trợ: {EMBEDDING_PROVIDER!r}. "
        "Dùng 'openai' hoặc 'sentence_transformers'."
    )


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        doc_type = "legal" if "legal" in path.parts else "news"
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Empty standardized document: {path}")

        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": path.stem,
                "doc_type": doc_type,
                "url": "",
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        for index, text in enumerate(splitter.split_text(document["content"])):
            content = text.strip()
            if not content:
                continue

            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": content,
                "metadata": {**document["metadata"], "chunk_index": index},
            })

    if not chunks:
        raise ValueError("No non-empty chunks were created")

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    # TODO: Embed theo batch và giữ nguyên các field của chunk.
    #
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    # TODO: Upsert ids, documents, embeddings và metadatas.
    #
    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    # Seed BM25 corpus của Task 6 bằng chính chunks này (cùng corpus với Task 5).
    from . import task6_lexical_search as task6

    task6.CORPUS = chunks
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()