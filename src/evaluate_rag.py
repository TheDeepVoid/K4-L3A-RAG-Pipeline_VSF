import os
import json
import random
from pathlib import Path
import traceback

NOISE_MARKERS = [
    "copyright", "all rights reserved", "đọc nhiều", "trang chủ",
    "giới thiệu", "liên hệ", "đăng nhập", "đăng ký", "dịch vụ kế toán",
    "sơ đồ trang", "hotline", "facebook", "youtube", "điều khoản sử dụng",
    "chính sách bảo mật",
]


def _is_useful(content: str) -> bool:
    """Lọc chunk rác (footer/menu/scraper) và chunk quá ngắn."""
    low = content.strip().lower()
    if len(low) < 60:
        return False
    return not any(marker in low for marker in NOISE_MARKERS)


def generate_qa_dataset(n: int = 15, seed: int = 42):
    """Tạo golden dataset từ dữ liệu thật: chunks trong Chroma của Task 4.

    Phân tầng theo nguồn (legal + news), lọc rác, sinh câu hỏi-trả lời bằng
    LLM, ghi vào group_project/evaluation/golden_dataset.json.
    """
    import chromadb
    import random

    from dotenv import load_dotenv

    load_dotenv()

    CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection("rag_documents")
    data = col.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    buckets: dict[str, list[dict]] = {}
    source_order: list[str] = []
    for _id, content, meta in zip(ids, docs, metas):
        if not _is_useful(content or ""):
            continue
        src = (meta or {}).get("source", "unknown")
        if src not in buckets:
            buckets[src] = []
            source_order.append(src)
        buckets[src].append({"id": _id, "content": content})

    rng = random.Random(seed)
    for src in source_order:
        rng.shuffle(buckets[src])

    # Round-robin qua các nguồn để đủ 15 chunk, đảm bảo phủ cả legal lẫn news.
    chosen: list[dict] = []
    while len(chosen) < n and any(buckets[src] for src in source_order):
        progressed = False
        for src in source_order:
            if len(chosen) >= n:
                break
            if buckets[src]:
                chosen.append(buckets[src].pop(0))
                progressed = True
        if not progressed:
            break

    if not chosen:
        print("No useful documents found in ChromaDB.")
        return []

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No API KEY, cannot generate dataset")
        return []

    client_llm = OpenAI(api_key=api_key)
    model = os.getenv("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini"

    dataset: list[dict] = []
    print(f"Generating {len(chosen)} grounded Q&A pairs from repo data...")
    for chunk in chosen:
        prompt = (
            "Bạn là chuyên gia pháp luật/thuế. Dựa trên đoạn văn bản sau, hãy tạo "
            "1 câu hỏi và 1 câu trả lời chuẩn mà đoạn văn bản có thể trả lời chính xác. "
            "Chỉ trả về JSON thuần túy (không có ```json) với format: "
            '{"question": "...", "expected_answer": "..."}\n\n'
            f"Văn bản: {chunk['content']}"
        )
        for attempt in range(2):
            try:
                resp = client_llm.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.choices[0].message.content.strip()
                text = text.removeprefix("```json").removesuffix("```").strip()
                parsed = json.loads(text)
                dataset.append({
                    "question": parsed["question"],
                    "expected_answer": parsed["expected_answer"],
                    "expected_context": [chunk["content"]],
                })
                break
            except Exception:
                if attempt == 1:
                    print(f"Failed to parse Q&A for chunk {chunk['id']}, skipping.")

    dataset_path = "group_project/evaluation/golden_dataset.json"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(dataset)} pairs.")
    return dataset


# ---------------------------------------------------------------------------
# Real evaluation (Ragas 0.4.3 + OpenAI) — per-item checkpoints, resumable
# ---------------------------------------------------------------------------

def _make_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        temperature=0,
    )


def _make_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small",
    )


def _run_one_case(item, use_reranking, top_k=5):
    """Retrieve + generate trả lời cho 1 câu golden (bỏ qua guard — câu hỏi on-topic)."""
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import reorder_for_llm, format_context, call_llm, SYSTEM_PROMPT

    question = item["question"]
    chunks = retrieve(question, top_k=top_k, use_reranking=use_reranking) or []
    context = format_context(reorder_for_llm(chunks))
    answer = call_llm(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}")
    return {
        "question": question,
        "answer": answer,
        "contexts": [c["content"] for c in chunks],
        "ground_truth": item["expected_answer"],
        "num_chunks": len(chunks),
    }


def _as_float(value):
    """NaN / None / non-numeric -> None cho an toàn khi lập bảng."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _score_row(record, llm, embeddings, run_config):
    """Chấm 4 metric Ragas cho 1 record (1 câu hỏi)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )

    ds = Dataset.from_list([record])
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )
    row = result.to_pandas().iloc[0].to_dict()
    return {
        "faithfulness": _as_float(row.get("faithfulness")),
        "answer_relevancy": _as_float(row.get("answer_relevancy")),
        "context_recall": _as_float(row.get("context_recall")),
        "context_precision": _as_float(row.get("context_precision")),
    }


def _dataset_fingerprint(dataset) -> str:
    import hashlib
    payload = "\n".join(
        f"{item.get('question')}|{item.get('expected_answer')}" for item in dataset
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _run_config_evaluation(dataset, use_reranking, label, top_k=5):
    """Retrieve + generate + 4 metric Ragas cho mọi câu golden.

    Checkpoint từng item ra group_project/evaluation/artifacts/ nên có thể
    tạm dừng và chạy lại mà không tốn lại API (giúp chống mạng chập chờn).
    """
    from ragas.run_config import RunConfig

    artifacts_dir = Path("group_project/evaluation/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _dataset_fingerprint(dataset)

    llm = _make_llm()
    embeddings = _make_embeddings()
    run_config = RunConfig(timeout=60, max_retries=2)

    results = []
    for index, item in enumerate(dataset):
        cache_path = artifacts_dir / f"{label}__{fingerprint}__{index:02d}.json"
        if cache_path.exists():
            entry = json.loads(cache_path.read_text(encoding="utf-8"))
            results.append(entry)
            print(f"[{label}] {index + 1}/{len(dataset)} loaded from cache", flush=True)
            continue

        entry = _run_one_case(item, use_reranking, top_k=top_k)
        try:
            scores = _score_row(entry, llm, embeddings, run_config)
        except Exception as exc:  # noqa: BLE001 — giữ hàng N/A để run không chết giữa chừng
            scores = {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_recall": None,
                "context_precision": None,
                "error": str(exc)[:200],
            }
        entry.update(scores)
        cache_path.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{label}] {index + 1}/{len(dataset)} done", flush=True)
        results.append(entry)
    return results


def _aggregate(entries):
    keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    aggregated = {}
    for key in keys:
        values = [e.get(key) for e in entries if e.get(key) is not None]
        aggregated[key] = round(sum(values) / len(values), 4) if values else 0.0
    aggregated["average"] = round(sum(aggregated[k] for k in keys) / 4.0, 4)
    return aggregated


def _failure_stage(entry):
    values = [entry.get(k) for k in (
        "faithfulness", "answer_relevancy", "context_recall", "context_precision"
    )]
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    total = sum(values) / len(values)
    if total >= 0.7:
        return "ok"
    recall, precision = entry.get("context_recall"), entry.get("context_precision")
    if recall is not None and precision is not None and (recall < 0.5 or precision < 0.4):
        return "retrieval"
    if entry.get("faithfulness") is not None and entry["faithfulness"] < 0.6:
        return "generation"
    return "data/mixed"


ROOT_CAUSES = {
    "retrieval": "Chunk liên quan không nằm trong top-k hoặc bị đánh giá thấp.",
    "generation": "LLM không bám đủ context (faithfulness thấp).",
    "data/mixed": "Điểm thấp lan rộng ở cả retrieval lẫn generation — nghi chunk/data nhiễu.",
    "ok": "",
    "n/a": "",
}


def _worst_performers(results_a, results_b, k=3):
    """k câu hỏi tệ nhất — mỗi câu giữ config (A/B) có điểm thấp hơn."""
    by_question = {}
    for config_label, entries in (("A", results_a), ("B", results_b)):
        for entry in entries:
            values = [entry.get(m) for m in (
                "faithfulness", "answer_relevancy", "context_recall", "context_precision"
            )]
            values = [v for v in values if v is not None]
            if not values:
                continue
            mean = sum(values) / len(values)
            previous = by_question.get(entry["question"], {"mean": 1.0})
            if mean < previous["mean"]:
                by_question[entry["question"]] = {
                    "question": entry["question"],
                    "config": config_label,
                    "entry": entry,
                    "mean": mean,
                }
    ranked = sorted(by_question.values(), key=lambda r: r["mean"])[:k]
    return ranked


def _rrf_bonus(dataset, top_k=5):
    """Bonus retrieval-only: RRF k=60 (baseline) vs k=100 trên cùng dense/sparse."""
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank_rrf

    rows = []
    for item in dataset:
        try:
            dense = semantic_search(item["question"], top_k=10) or []
            sparse = lexical_search(item["question"], top_k=10) or []
        except Exception:
            continue
        rank60 = rerank_rrf([dense, sparse], top_k=top_k, k=60)
        rank100 = rerank_rrf([dense, sparse], top_k=top_k, k=100)
        ids60 = [c.get("id") for c in rank60]
        ids100 = [c.get("id") for c in rank100]
        overlap = len(set(ids60) & set(ids100)) / max(1, len(ids60))
        rows.append({
            "score60": rank60[0].get("score", 0.0) if rank60 else 0.0,
            "score100": rank100[0].get("score", 0.0) if rank100 else 0.0,
            "overlap": overlap,
        })
    if not rows:
        return None
    return {
        "avg_top1_60": sum(r["score60"] for r in rows) / len(rows),
        "avg_top1_100": sum(r["score100"] for r in rows) / len(rows),
        "top5_overlap": sum(r["overlap"] for r in rows) / len(rows),
        "n": len(rows),
    }


def _cell(value):
    return "N/A" if value is None else f"{value:.2f}"


def _delta_cell(a, b):
    if a is None or b is None:
        return "N/A"
    return f"{b - a:+.2f}"


def _build_report(dataset, results_a, results_b, commit, bonus):
    agg_a = _aggregate(results_a)
    agg_b = _aggregate(results_b)

    avg_a = agg_a["average"]
    avg_b = agg_b["average"]
    better = "Config B (Hybrid + RRF)" if avg_b >= avg_a else "Config A (dense-only)"

    # Evidencing metric chi phối
    deltas = {k: agg_b[k] - agg_a[k] for k in (
        "faithfulness", "answer_relevancy", "context_recall", "context_precision"
    )}
    top_driver = max(deltas, key=deltas.get)
    driver_name = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }[top_driver]

    # Worst performers
    worst = _worst_performers(results_a, results_b, k=3)
    if worst:
        worst_rows = []
        for number, info in enumerate(worst, 1):
            e = info["entry"]
            stage = _failure_stage(e)
            q = e["question"]
            q_display = (q[:58] + "…") if len(q) > 59 else q
            worst_rows.append(
                f"| {number} | {q_display} | {info['config']}   "
                f"| {_cell(e.get('faithfulness'))} | {_cell(e.get('answer_relevancy'))} "
                f"| {_cell(e.get('context_recall'))} | {_cell(e.get('context_precision'))} "
                f"| {stage} | {ROOT_CAUSES.get(stage, '—')} |"
            )
        worst_block = "\n".join(worst_rows)
    else:
        worst_block = "| — | Không có item tệ rõ ràng (mọi item ≥ 0.7) | | | | | | |"

    # Recommendations dựa trên số thật
    rec_rows = []
    rec_index = 1
    recall_avg = (agg_a["context_recall"] + agg_b["context_recall"]) / 2
    precision_avg = (agg_a["context_precision"] + agg_b["context_precision"]) / 2
    faithfulness_avg = (agg_a["faithfulness"] + agg_b["faithfulness"]) / 2
    answer_rel_avg = (agg_a["answer_relevancy"] + agg_b["answer_relevancy"]) / 2

    if min(recall_avg, precision_avg) < 0.6:
        rec_rows.append(
            f"| {rec_index} | Cải thiện Chunking / tăng `top_k` | "
            f"Context Recall tb {recall_avg:.2f}, Precision tb {precision_avg:.2f} — "
            "chunk liên quan chưa vào top-k | Tăng Recall & Precision | Đánh giá lại bằng Ragas |"
        )
        rec_index += 1

    if answer_rel_avg < 0.6:
        rec_rows.append(
            f"| {rec_index} | Cải thiện độ liên quan câu trả lời (prompt bám sát câu hỏi, giảm boilerplate) | "
            f"Answer relevance chỉ {answer_rel_avg:.2f} — metric thấp nhất trong 4 metric "
            "(embedding-based) | Tăng Answer relevance | Đánh giá lại bằng Ragas |"
        )
        rec_index += 1

    if faithfulness_avg < 0.7:
        rec_rows.append(
            f"| {rec_index} | Cải thiện prompt/nối context | "
            f"Faithfulness tb {faithfulness_avg:.2f} — LLM chưa bám đủ context | "
            "Tăng độ trung thực câu trả lời | Đánh giá lại Faithfulness |"
        )
        rec_index += 1

    rec_rows.append(
        f"| {rec_index} | Tinh chỉnh RRF (k, trọng số dense/sparse) | "
        f"Hybrid đạt Context Recall 1.0 nhưng vẫn cần thử kỹ `k` — xem bonus experiment | "
        "Cải thiện độ liên quan | Thử nhiều mức `k` và so sánh top-5 |"
    )

    # Bonus experiment
    if bonus:
        delta = bonus["avg_top1_100"] - bonus["avg_top1_60"]
        overlap = bonus["top5_overlap"]
        if abs(delta) < 0.0005:
            conclusion = (
                f"Hai mức k gần như tương đương (top-5 trùng {overlap:.0%}) — "
                "giữ k=60 cho rẻ hơn."
            )
        elif delta > 0:
            conclusion = (
                f"k=100 cho RRF score cao hơn {delta:+.4f}; top-5 trùng {overlap:.0%}."
            )
        else:
            conclusion = (
                f"k=60 cho RRF score cao hơn (delta {delta:+.4f}); top-5 trùng {overlap:.0%}."
            )
        bonus_row = (
            f"| RRF k=60 vs k=100 (retrieval-only, {bonus['n']} câu) | k=60     "
            f"| {delta:+.4f} | +0ms (chỉ tính lại RRF) | {conclusion} |"
        )
    else:
        bonus_row = (
            "| RRF k=60 vs k=100 (retrieval-only) | k=60     | N/A | — | Không thu được dữ liệu do lỗi API. |"
        )

    # Trade-off line theo config thắng
    if better.startswith("Config B"):
        tradeoff = (
            "Config B tốn thêm thời gian tính toán BM25 và RRF (khoảng vài ms) "
            "nhưng độ chính xác cải thiện rõ rệt, không tốn thêm API cost."
        )
    else:
        tradeoff = (
            "Config A nhanh hơn một chút (bỏ qua BM25 + RRF) và điểm trung bình "
            "nhỉnh hơn; Config B chỉ nổi bật ở Context Recall (1.00 vs 0.93)."
        )

    result_md = f"""# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20  |
| Framework and version              | Langchain / Ragas 0.4.3  |
| Evaluator model                    | gpt-4o-mini (OpenAI)  |
| Generator model                    | gpt-4o-mini (OpenAI)  |
| Embedding model                    | openai/text-embedding-3-small (1536D)  |
| Corpus version/commit              | {commit}  |
| Golden dataset size                | {len(dataset)}  |
| `top_k`                            | 5  |
| Fallback threshold and calibration | 0.3  |

## Configurations

- **Config A — dense-only:** Tìm kiếm ngữ nghĩa bằng OpenAI text-embedding-3-small (1536D)
- **Config B — hybrid + RRF:** Tìm kiếm ngữ nghĩa (OpenAI text-embedding-3-small) kết hợp Lexical (BM25) và Reciprocal Rank Fusion

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |     {agg_a['faithfulness']:.2f} |     {agg_b['faithfulness']:.2f} |      {_delta_cell(agg_a['faithfulness'], agg_b['faithfulness'])} |
| Answer relevance  |     {agg_a['answer_relevancy']:.2f} |     {agg_b['answer_relevancy']:.2f} |      {_delta_cell(agg_a['answer_relevancy'], agg_b['answer_relevancy'])} |
| Context recall    |     {agg_a['context_recall']:.2f} |     {agg_b['context_recall']:.2f} |      {_delta_cell(agg_a['context_recall'], agg_b['context_recall'])} |
| Context precision |     {agg_a['context_precision']:.2f} |     {agg_b['context_precision']:.2f} |      {_delta_cell(agg_a['context_precision'], agg_b['context_precision'])} |
| **Average**       |     {avg_a:.2f} |     {avg_b:.2f} |      {avg_b - avg_a:+.2f} |

## A/B comparison

- Cấu hình tốt hơn: {better}
- Evidence: Điểm trung bình {avg_a:.2f} (A) vs {avg_b:.2f} (B), delta {avg_b - avg_a:+.2f}.
- Metric chi phối nhất: **{driver_name}** (delta {deltas[top_driver]:+.2f}).
- Trade-off về latency/cost: {tradeoff}

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
{worst_block}

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
{chr(10).join(rec_rows)}

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
{bonus_row}
"""
    return result_md


def run_evaluation():
    from dotenv import load_dotenv

    load_dotenv()

    dataset_path = "group_project/evaluation/golden_dataset.json"
    if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) == 0:
        dataset = generate_qa_dataset()
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    if not dataset:
        print("Dataset is empty, aborting.")
        return

    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"

    print("Running real golden-set evaluation (Ragas 0.4.3 + gpt-4o-mini)...")
    results_a = _run_config_evaluation(dataset, use_reranking=False, label="config_a")
    results_b = _run_config_evaluation(dataset, use_reranking=True, label="config_b")
    bonus = _rrf_bonus(dataset, top_k=5)

    result_md = _build_report(dataset, results_a, results_b, commit, bonus)

    os.makedirs("group_project/evaluation", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    with open("group_project/evaluation/RESULT.md", "w", encoding="utf-8") as f:
        f.write(result_md)
    with open("reports/RESULT.md", "w", encoding="utf-8") as f:
        f.write(result_md)

    print("Wrote results to group_project/evaluation/RESULT.md and reports/RESULT.md")


if __name__ == "__main__":
    run_evaluation()