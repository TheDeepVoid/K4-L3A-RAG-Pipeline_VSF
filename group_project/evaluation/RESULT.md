# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20  |
| Framework and version              | Langchain / Ragas 0.4.3  |
| Evaluator model                    | gpt-4o-mini (OpenAI)  |
| Generator model                    | gpt-4o-mini (OpenAI)  |
| Embedding model                    | openai/text-embedding-3-small (1536D)  |
| Corpus version/commit              | c7a578b  |
| Golden dataset size                | 15  |
| `top_k`                            | 5  |
| Fallback threshold and calibration | 0.3  |

## Configurations

- **Config A — dense-only:** Tìm kiếm ngữ nghĩa bằng OpenAI text-embedding-3-small (1536D)
- **Config B — hybrid + RRF:** Tìm kiếm ngữ nghĩa (OpenAI text-embedding-3-small) kết hợp Lexical (BM25) và Reciprocal Rank Fusion

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |     0.97 |     0.85 |      -0.12 |
| Answer relevance  |     0.45 |     0.43 |      -0.01 |
| Context recall    |     0.93 |     1.00 |      +0.07 |
| Context precision |     0.94 |     0.93 |      -0.01 |
| **Average**       |     0.82 |     0.80 |      -0.02 |

## A/B comparison

- Cấu hình tốt hơn: Config A (dense-only)
- Evidence: Điểm trung bình 0.82 (A) vs 0.80 (B), delta -0.02.
- Metric chi phối nhất: **Context Recall** (delta +0.07).
- Trade-off về latency/cost: Config A nhanh hơn một chút (bỏ qua BM25 + RRF) và điểm trung bình nhỉnh hơn; Config B chỉ nổi bật ở Context Recall (1.00 vs 0.93).

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
| 1 | Số tiền thuế, khoản thu khác, tiền chậm nộp, tiền phạt nộp… | A   | 0.60 | 0.56 | 0.00 | 0.87 | retrieval | Chunk liên quan không nằm trong top-k hoặc bị đánh giá thấp. |
| 2 | Chủ quản nền tảng thương mại điện tử nước ngoài có trách n… | B   | 0.50 | 0.00 | 1.00 | 1.00 | generation | LLM không bám đủ context (faithfulness thấp). |
| 3 | Khi nào thuế khoán sẽ chính thức bị xóa bỏ theo thông tin … | B   | 0.50 | 0.36 | 1.00 | 0.80 | generation | LLM không bám đủ context (faithfulness thấp). |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
| 1 | Cải thiện độ liên quan câu trả lời (prompt bám sát câu hỏi, giảm boilerplate) | Answer relevance chỉ 0.44 — metric thấp nhất trong 4 metric (embedding-based) | Tăng Answer relevance | Đánh giá lại bằng Ragas |
| 2 | Tinh chỉnh RRF (k, trọng số dense/sparse) | Hybrid đạt Context Recall 1.0 nhưng vẫn cần thử kỹ `k` — xem bonus experiment | Cải thiện độ liên quan | Thử nhiều mức `k` và so sánh top-5 |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| RRF k=60 vs k=100 (retrieval-only, 15 câu) | k=60     | -0.0129 | +0ms (chỉ tính lại RRF) | k=60 cho RRF score cao hơn (delta -0.0129); top-5 trùng 100%. |
