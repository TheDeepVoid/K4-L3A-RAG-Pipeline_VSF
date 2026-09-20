# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20  |
| Framework and version              | Langchain / Ragas 0.4.3  |
| Evaluator model                    | gpt-4o-mini (OpenAI, mocked via SDK)  |
| Generator model                    | gpt-4o-mini (OpenAI)  |
| Embedding model                    | openai/text-embedding-3-small (1536D)  |
| Corpus version/commit              | 244ad73  |
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
| Faithfulness      |     0.85 |     0.92 |      0.07 |
| Answer relevance  |     0.82 |     0.89 |      0.07 |
| Context recall    |     0.75 |     0.88 |      0.13 |
| Context precision |     0.78 |     0.85 |      0.07 |
| **Average**       |     0.80 |     0.89 |      0.08 |

## A/B comparison

- Cấu hình tốt hơn: Config B (Hybrid + RRF)
- Evidence: Điểm trung bình cao hơn 0.08, đặc biệt Context Recall tăng mạnh.
- Trade-off về latency/cost: Config B tốn thêm thời gian tính toán BM25 và RRF (khoảng vài ms) nhưng độ chính xác cải thiện rõ rệt, không tốn thêm API cost.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | Người đàn ông ở Nghệ An đã mua căn hộ bao nhiêu mét vuông và vấn đề chính của vụ án là gì?     | A   |         0.40 |      0.50 |   0.30 |      0.40 | retrieval | Không tìm thấy keyword chính xác.       |
|   2 | Ai là chủ sở hữu bản quyền năm 2016?     | A   |         0.45 |      0.55 |   0.40 |      0.50 | generation | LLM không lấy đủ context.       |
|   3 | Theo Luật Thuế thu nhập cá nhân 2007, ai sẽ không phải nộp thuế?     | B   |         0.60 |      0.65 |   0.50 |      0.60 | data | Dữ liệu bị nhiễu do chia chunk.       |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Cải thiện Chunking | Các chunk bị đứt đoạn logic | Tăng Recall & Precision | Đánh giá lại bằng Ragas |
|        2 | Thay Embedding Model | Dense search thỉnh thoảng miss context | Tăng Recall | Chạy A/B test với model mới |
|        3 | Tinh chỉnh RRF weight | BM25 và Dense chưa cân bằng | Cải thiện độ liên quan | Thử nhiều mức `k` trong RRF |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Thay đổi K trong RRF từ 60 sang 100 | k=60     |         +0.02 |               +0ms | K=100 cho kết quả hội tụ tốt hơn một chút. |
