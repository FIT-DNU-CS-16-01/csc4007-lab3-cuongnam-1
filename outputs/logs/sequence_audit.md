# Sequence Audit

- n_train: 20000
- vocab_size: 20000
- max_len: 256
- orig_len_median: 196.00
- orig_len_p95: 665.00
- truncation_rate: 0.3475
- unk_rate: 0.0238
- avg_pad_ratio: 0.2537

## Gợi ý đọc kết quả
- Nếu truncation_rate cao, `max_len` có thể đang quá nhỏ.
- Nếu avg_pad_ratio quá cao, `max_len` có thể đang quá lớn.
- Nếu unk_rate cao, vocab_size có thể đang quá nhỏ hoặc tokenization chưa phù hợp.
