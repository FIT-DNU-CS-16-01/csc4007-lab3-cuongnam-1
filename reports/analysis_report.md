# CSC4007 — Lab 3 Analysis Report (RNN + W&B)

## 1. Thông tin sinh viên
- Họ và tên:
- Mã sinh viên:
- Lớp:
- Repo GitHub:
- W&B project:
- Tên run tốt nhất:

## 2. Mục tiêu thí nghiệm
Viết 3–5 dòng trả lời các câu hỏi sau:
- Lab 3 khác Lab 2 ở điểm nào?
- Vì sao cần chuyển từ BoW/TF-IDF sang mô hình chuỗi?
- Bạn kỳ vọng RNN cải thiện điều gì trên IMDB?

## 3. Sequence audit
Dựa trên `outputs/logs/sequence_audit.md`, nêu ít nhất 3 nhận xét có số liệu hoặc bằng chứng cụ thể.

1.
2.
3.

Gợi ý:
- Review có độ dài phân bố như thế nào?
- `max_len` bạn chọn có hợp lý không?
- Có nhiều review bị cắt ngắn không?
- Điều này ảnh hưởng thế nào đến bài toán sentiment classification?

## 4. Thiết lập mô hình và huấn luyện
Ghi lại cấu hình tốt nhất của bạn:

- vocab_size:
- max_len:
- embed_dim:
- hidden_dim:
- batch_size:
- epochs:
- learning rate:
- dropout:
- seed:
- early stopping patience:
- wandb_mode:

Giải thích ngắn gọn vì sao bạn chọn cấu hình này.

## 5. Baseline ML vs RNN
Điền bảng dựa trên `outputs/metrics/baseline_vs_rnn.csv`.

| Mô hình | Accuracy | Macro-F1 | Ghi chú |
|---|---:|---:|---|
| Baseline ML (Lab 2) |  |  |  |
| RNN (Lab 3) |  |  |  |

### Nhận xét (5–7 dòng)
Trả lời:
- RNN có tốt hơn baseline hay không?
- Nếu tốt hơn, cải thiện đó có đáng kể không?
- Nếu chưa tốt hơn, nguyên nhân hợp lý là gì?
- Vai trò của thứ tự từ trong bài toán IMDB thể hiện ra sao?

## 6. Learning curves và W&B
Đính kèm hoặc chèn:
- `outputs/figures/loss_curve.png`
- `outputs/figures/metric_curve.png`
- hoặc ảnh chụp dashboard W&B

Trả lời ngắn các câu hỏi sau:
- Epoch tốt nhất là epoch nào?
- Có dấu hiệu overfitting không?
- W&B giúp bạn quan sát điều gì rõ hơn so với chỉ đọc terminal log?
- Bạn có so sánh ít nhất 2 run không? Nếu có, run nào tốt hơn và vì sao?

## 7. Error analysis (ít nhất 10 mẫu sai)
Dựa trên `outputs/error_analysis/error_analysis.csv`, chọn và phân tích ít nhất 10 mẫu dự đoán sai.

### Gợi ý nhóm lỗi
- phủ định;
- mixed sentiment;
- review dài;
- sarcasm/irony;
- mô hình rất tự tin nhưng vẫn sai;
- câu có nhiều chuyển ý hoặc phụ thuộc ngữ cảnh xa.

### Tổng hợp lỗi
1.
2.
3.

### Ví dụ bảng ghi nhận lỗi
| ID | True label | Pred label | Vì sao sai? | Hướng cải thiện |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

## 8. Bài học rút ra
Viết 5–7 dòng về những điều bạn học được khi chuyển từ TF-IDF/LogReg sang Embedding + RNN.

Có thể đề cập:
- ưu điểm và hạn chế của RNN;
- vai trò của sequence length;
- tầm quan trọng của validation set;
- ý nghĩa của learning curves;
- lợi ích của W&B trong việc theo dõi thí nghiệm.

## 9. Tự đánh giá theo rubric
Sinh viên tự chấm sơ bộ theo `reports/rubric.md` trước khi nộp bài.
