# CSC4007 — Lab 3 Rubric (RNN + W&B)

Tổng điểm: **10 điểm**

## 1. Repo chạy được và artefact đầy đủ (2.0 điểm)
- **2.0**: Repo chạy được; có đầy đủ file output quan trọng; cấu trúc rõ ràng; không lỗi đường dẫn cơ bản.
- **1.0**: Chạy được nhưng thiếu một số artefact hoặc output chưa nhất quán.
- **0.0**: Repo không chạy được hoặc thiếu phần lớn artefact.

## 2. Mô hình RNN và quy trình huấn luyện đúng (2.0 điểm)
- **2.0**: Dùng đúng pipeline sequence; có embedding + RNN; có validation; có early stopping hoặc cơ chế kiểm soát overfitting; seed rõ ràng.
- **1.0**: Mô hình chạy được nhưng quy trình huấn luyện chưa chặt chẽ hoặc cấu hình thiếu rõ ràng.
- **0.0**: Mô hình sai logic hoặc không huấn luyện được.

## 3. Sử dụng W&B hợp lý (1.5 điểm)
- **1.5**: Có log run rõ ràng; ghi lại hyperparameters và metric theo epoch; biết dùng W&B để so sánh run.
- **0.75**: Có dùng W&B nhưng còn sơ sài, thiếu một phần log quan trọng.
- **0.0**: Không dùng W&B hoặc không có bằng chứng sử dụng.

## 4. So sánh baseline ML vs RNN (1.5 điểm)
- **1.5**: Có bảng so sánh rõ ràng giữa Lab 2 và Lab 3; diễn giải hợp lý, không kết luận cảm tính.
- **0.75**: Có bảng so sánh nhưng nhận xét ngắn hoặc chưa dựa nhiều trên bằng chứng.
- **0.0**: Không có so sánh hoặc so sánh sai.

## 5. Learning curves và diễn giải kết quả (1.5 điểm)
- **1.5**: Có loss/metric curves; nhận xét được epoch tốt nhất, dấu hiệu overfitting/underfitting.
- **0.75**: Có hình nhưng diễn giải còn mờ nhạt.
- **0.0**: Không có learning curves hoặc không phân tích.

## 6. Error analysis (1.5 điểm)
- **1.5**: Phân tích ít nhất 10 mẫu sai; nhóm lỗi rõ; có hướng cải thiện hợp lý.
- **0.75**: Có error analysis nhưng còn nông hoặc chưa đủ số lượng/chất lượng.
- **0.0**: Không có error analysis.

## Ghi chú trừ điểm thường gặp
- Chỉ báo cáo accuracy mà không quan tâm macro-F1 hoặc confusion matrix.
- Không nêu cấu hình mô hình.
- Không ghi lại run W&B.
- Không phân biệt được train/validation/test.
- Kết luận "RNN tốt hơn" nhưng không có số liệu chứng minh.
