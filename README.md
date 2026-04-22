[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/bHCXxsD5)
# CSC4007 — Lab 3: Sequence Models with RNN + Weights & Biases

## Giới thiệu bài thực hành

Sau Lab 2, sinh viên đã xây dựng được một pipeline NLP cơ bản theo hướng:

- kiểm tra dữ liệu,
- tiền xử lý văn bản,
- biểu diễn đặc trưng bằng BoW/TF-IDF,
- huấn luyện mô hình baseline kiểu Logistic Regression hoặc Linear SVM,
- đánh giá bằng confusion matrix, macro-F1, và error analysis.

Tuy nhiên, các mô hình ở Lab 2 xem văn bản chủ yếu như một tập đặc trưng rời rạc. Ở Lab 3, sinh viên chuyển sang học một hướng tiếp cận khác: mô hình hóa **chuỗi từ theo thứ tự xuất hiện**. Đây là lý do RNN xuất hiện trong bài lab này.

RNN giúp mô hình khai thác thông tin tuần tự của câu, đặc biệt hữu ích khi ý nghĩa phụ thuộc vào vị trí và ngữ cảnh, ví dụ các hiện tượng phủ định, chuyển ý, hoặc sắc thái cảm xúc thay đổi dọc theo review. Lab này cũng là bước chuẩn bị cần thiết trước khi học LSTM/GRU và các mô hình chuỗi mạnh hơn ở các buổi sau.

Bên cạnh đó, sinh viên sẽ làm quen với **Weights & Biases (W&B)** để theo dõi thí nghiệm học máy một cách có hệ thống. Thay vì chỉ nhìn log ở terminal, sinh viên sẽ ghi lại hyperparameters, learning curves, và kết quả của nhiều lần chạy để so sánh rõ ràng hơn.

## Mục tiêu bài thực hành

Sau bài lab này, sinh viên cần:

1. Biểu diễn văn bản dưới dạng **chuỗi token** thay vì vector BoW/TF-IDF.
2. Xây dựng và huấn luyện mô hình **Embedding + RNN** cho bài toán phân loại cảm xúc trên IMDB.
3. Hiểu vai trò của các thành phần: vocabulary, padding, sequence length, embedding, hidden state, dropout, early stopping.
4. Sử dụng **W&B** để theo dõi các lần chạy và quan sát learning curves.
5. So sánh kết quả **baseline ML của Lab 2** với **RNN của Lab 3**.
6. Phân tích lỗi mô hình trên các mẫu dự đoán sai, thay vì chỉ nhìn accuracy.

## Mô tả ngắn về dataset

Bài lab tiếp tục sử dụng **IMDB** cho phân loại cảm xúc review phim với hai nhãn:

- `positive`
- `negative`

Dataset mặc định được tải qua thư viện `datasets` của Hugging Face.

Trong phiên bản đã chỉnh của starter kit, đường chạy với IMDB giữ đúng tinh thần chuẩn hơn:

- dùng **split gốc** của IMDB;
- tạo **validation set từ train split**;
- giữ **test split gốc** để đánh giá cuối cùng;
- chỉ dùng `sample_imdb_tiny.csv` cho smoke test cục bộ và CI nhanh.

## Giới thiệu ngắn về RNN

Trong Lab 2, văn bản được đổi thành các vector đặc trưng tĩnh như BoW hoặc TF-IDF. Các cách này mạnh, đơn giản, và rất phù hợp làm baseline, nhưng gần như không biểu diễn trực tiếp được **thứ tự xuất hiện của từ**.

RNN (Recurrent Neural Network) xử lý dữ liệu theo từng bước thời gian. Với bài toán text classification, mô hình đọc lần lượt từng token trong câu, cập nhật trạng thái ẩn, rồi dùng trạng thái đó để đưa ra dự đoán cuối cùng. Trong repo này, mô hình cơ bản gồm:

- tầng `Embedding` để ánh xạ token ID sang vector dày đặc,
- tầng `RNN` để xử lý chuỗi,
- tầng phân loại để dự đoán nhãn đầu ra.

Sinh viên cần đặc biệt chú ý các điểm sau:

- **padding**: các câu phải có cùng độ dài trong một batch;
- **max_len**: nếu quá ngắn thì mất thông tin, quá dài thì tốn chi phí và dễ nhiễu;
- **dropout**: giúp giảm overfitting;
- **early stopping**: dừng sớm khi mô hình không còn cải thiện trên validation set;
- **seed**: cần cố định để kết quả có thể tái lập.

## Giới thiệu ngắn về Weights & Biases (W&B)

Weights & Biases là công cụ hỗ trợ theo dõi thí nghiệm học máy. Trong bài lab này, W&B được dùng để:

- lưu lại hyperparameters của từng lần chạy;
- theo dõi `train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1` theo epoch;
- so sánh nhiều run với nhau;
- hỗ trợ sinh viên đọc learning curves và phát hiện overfitting.

Repo hỗ trợ cả hai chế độ:

- `online`: log lên tài khoản W&B;
- `offline`: lưu log cục bộ, phù hợp khi không muốn đăng nhập hoặc khi chạy CI.

## Nội dung thực hành

### Chuẩn bị

Sinh viên cần:

- có tài khoản GitHub;
- có môi trường Python/conda dùng cho học phần;
- đã hoàn thành Lab 2 hoặc ít nhất có thể đọc và hiểu kết quả baseline ML ở Lab 2;
- có tài khoản W&B nếu muốn log online.

## Fork starter kit về GitHub cá nhân (BẮT BUỘC)

Giảng viên đã chuẩn bị sẵn starter repo cho bài lab này. Sinh viên thực hiện:

1. Mở repo starter kit của Lab 3.
2. Bấm **Fork** để tạo bản sao về GitHub cá nhân.
3. Sau đó repo của sinh viên sẽ có dạng:

```text
https://github.com/<username>/<repo-name>
```

## Clone repo về máy

Mở **Anaconda Prompt** (Windows) hoặc **Terminal** (macOS/Linux), chạy:

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

### Kích hoạt môi trường

```bash
conda activate csc4007-nlp
```

hoặc nếu dùng `venv`:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

### Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu muốn log online với W&B:

```bash
wandb login
```

## Chạy bài lab

### Cách 1 — Chạy với IMDB thật

```bash
python run_lab3.py \
  --dataset imdb \
  --seed 42 \
  --vocab_size 20000 \
  --max_len 256 \
  --embed_dim 128 \
  --hidden_dim 128 \
  --batch_size 64 \
  --epochs 6 \
  --lr 1e-3 \
  --dropout 0.3 \
  --use_wandb
```

Khi chạy với IMDB:

- `train` và `test` lấy từ split gốc của dataset;
- `val` được tách ra từ `train` theo seed;
- nếu truyền `--max_rows`, repo chỉ lấy **một phần nhỏ của split gốc** để test nhanh, thay vì trộn toàn bộ dataset rồi chia lại.

### Cách 2 — Chạy smoke test với dữ liệu nhỏ

```bash
python run_lab3.py \
  --dataset local_csv \
  --data_path data/raw/sample_imdb_tiny.csv \
  --epochs 2 \
  --batch_size 8 \
  --max_len 64 \
  --vocab_size 2000 \
  --embed_dim 32 \
  --hidden_dim 32 \
  --wandb_mode offline \
  --use_wandb
```

## Cấu trúc repo

```text
csc4007_lab3_starter_aligned/
├── .github/workflows/ci.yml
├── data/
│   └── raw/
│       ├── README.md
│       └── sample_imdb_tiny.csv
├── notebooks/
│   └── README.md
├── outputs/
├── reports/
│   ├── analysis_report.md
│   └── rubric.md
├── requirements.txt
├── run_lab3.py
└── src/
    ├── data.py
    ├── error_analysis.py
    ├── evaluate.py
    ├── model.py
    ├── sequence_audit.py
    ├── train.py
    ├── utils.py
    └── wandb_utils.py
```

## Ý nghĩa các output

Sau khi chạy xong, repo sẽ sinh ra các file quan trọng sau:

- `outputs/logs/sequence_audit.md`: thống kê cơ bản về độ dài chuỗi, tỷ lệ cắt ngắn, phân bố nhãn;
- `outputs/metrics/epoch_history.csv`: metric theo từng epoch;
- `outputs/metrics/metrics_summary.md`: kết quả tổng hợp của mô hình tốt nhất;
- `outputs/metrics/baseline_vs_rnn.csv`: bảng so sánh baseline Lab 2 với RNN Lab 3;
- `outputs/figures/loss_curve.png`: đường train/validation loss;
- `outputs/figures/metric_curve.png`: đường accuracy/F1 theo epoch;
- `outputs/figures/confusion_matrix.png`: ma trận nhầm lẫn trên test set;
- `outputs/error_analysis/error_analysis.csv`: danh sách các mẫu sai để phân tích;
- `outputs/error_analysis/error_analysis_summary.md`: tóm tắt nhóm lỗi thường gặp;
- `outputs/models/best_model.pt`: trọng số mô hình tốt nhất;
- `outputs/predictions/test_predictions.csv`: dự đoán trên test set;
- `outputs/logs/run_summary.json`: tóm tắt thông số và đường dẫn output của lần chạy.

## Yêu cầu sinh viên phải thực hiện

1. Chạy thành công mô hình **Embedding + RNN** trên IMDB.
2. Sử dụng **W&B** để log ít nhất một lần chạy hoàn chỉnh.
3. Thử **ít nhất 2 cấu hình khác nhau**. Có thể thay đổi một hoặc nhiều tham số sau:
   - `max_len`
   - `hidden_dim`
   - `dropout`
   - `lr`
   - `batch_size`
   - `early stopping patience`
4. Hoàn thành so sánh giữa **baseline ML của Lab 2** và **RNN của Lab 3**.
5. Phân tích **ít nhất 10 mẫu sai** trong file error analysis.
6. Điền đầy đủ `reports/analysis_report.md`.

## Cách nối với kết quả Lab 2

Nếu sinh viên đã có file `metrics_summary.json` của Lab 2, có thể truyền vào như sau:

```bash
python run_lab3.py --baseline_metrics_path /path/to/lab2/outputs/metrics/metrics_summary.json
```

Khi đó repo sẽ tạo bảng `baseline_vs_rnn.csv` để phục vụ phần so sánh trong báo cáo.

## Một số lỗi thường gặp

Các lỗi phổ biến trong bài lab này gồm:

- padding hoặc sequence length chưa hợp lý;
- tensor sai kích thước batch/sequence;
- quên cố định seed;
- chọn mô hình quá lớn khiến overfit;
- quên early stopping;
- chỉ nhìn accuracy mà bỏ qua macro-F1 và confusion matrix;
- chạy W&B nhưng không ghi lại tên run hoặc không so sánh các run.

## Checklist nộp bài

Sinh viên cần nộp repo GitHub cá nhân, trong đó có tối thiểu:

- mã nguồn đã hoàn thiện;
- `reports/analysis_report.md` đã điền nội dung;
- các file output cần thiết trong `outputs/`;
- ảnh learning curves;
- confusion matrix;
- bảng so sánh baseline vs RNN;
- file error analysis;
- nếu dùng W&B online: ghi rõ link dashboard hoặc tên project/run trong báo cáo.

## Rubric chấm bài

Rubric chi tiết nằm tại:

- `reports/rubric.md`

- `reports/rubric.md`

## CI dùng để làm gì?

Repo hiện có **2 workflow**:

- `lab3-ci`: smoke test nhanh với `sample_imdb_tiny.csv`;
- `lab3-imdb-ci`: smoke test riêng cho đường chạy `--dataset imdb` với `--max_rows 200`.

Workflow IMDB giúp kiểm tra rằng repo GitHub vẫn chạy được với dữ liệu thật, trong khi workflow local giúp kiểm tra nhanh hơn ở mỗi lần sửa code. Cả hai workflow đều bật W&B ở `offline` mode và kiểm tra các artefact bắt buộc sau khi chạy xong.

CI giúp sinh viên biết repo có chạy được hay không, nhưng **không thay thế cho việc hoàn thành đầy đủ phân tích học thuật trong báo cáo**.
