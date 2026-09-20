"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
from pathlib import Path

from dotenv import load_dotenv


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 100

COLLECTION_NAME = "rag_documents"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Tạo embedding cho các đoạn văn bằng OpenAI Embeddings API."""
    if not texts:
        return []

    load_dotenv()

    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    if provider != "openai":
        raise ValueError(
            f"Task 4 is configured for OpenAI embeddings, got provider: {provider}"
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in the project .env file."
        )

    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    vectors: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(
            model=model_name,
            input=batch,
        )

        batch_data = sorted(response.data, key=lambda item: item.index)
        batch_vectors = [item.embedding for item in batch_data]

        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                "OpenAI returned a different number of embeddings than inputs."
            )

        vectors.extend(batch_vectors)

    if len(vectors) != len(texts):
        raise RuntimeError(
            "Embedding count does not match input text count."
        )

    for vector in vectors:
        if len(vector) != EMBEDDING_DIM:
            raise RuntimeError(
                f"Expected embedding dimension {EMBEDDING_DIM}, "
                f"got {len(vector)}."
            )

    return vectors


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    # TODO: Tạo hoặc mở persistent collection.
    #
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    # raise NotImplementedError("Implement get_collection")


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    # TODO: Đọc mọi .md và tạo Document theo contract.
    #
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

    if not documents:
        raise ValueError(f"No documents found in {STANDARDIZED_DIR}")
    
    return documents
    # raise NotImplementedError("Implement load_documents")


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    # TODO: Chunk bằng RecursiveCharacterTextSplitter.
    #
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
    # raise NotImplementedError("Implement chunk_documents")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    if not chunks:
        return []

    vectors = embed_texts([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks
    # raise NotImplementedError("Implement embed_chunks")


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        raise ValueError("Cannot index an empty chunk list")

    ids = [chunk["id"] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs detected")

    metadatas = []
    for chunk in chunks:
        metadata = {
            key: "" if value is None else value
            for key, value in chunk["metadata"].items()
        }
        metadatas.append(metadata)

    collection = get_collection()
    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=metadatas,
    )
    # raise NotImplementedError("Implement index_to_vectorstore")


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
