# Báo Cáo Kỹ Thuật: Tối Ưu Hóa Chi Phí GPU (GPU FinOps Optimization Report)
**Học viên:** Nguyễn Việt Thắng
**Mã sinh viên:** 2A202601321
**Trạng thái hệ thống:** 11/11 Automated Checks Passed · 15/15 Pytest Passed  

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

NimbusAI là một công ty khởi nghiệp AI đang phát triển hệ thống LLM phục vụ đa tác vụ (Chat, Search, RAG, Assistant, Fine-tuning). Trước dự án tối ưu hóa, chi phí hạ tầng GPU hàng tháng của công ty ở mức **$27,133/tháng** với đơn giá phục vụ suy luận là **$6.488 / 1 triệu token**.

Bằng việc áp dụng toàn diện các nguyên lý **GPU FinOps**, chúng tôi đã cắt giảm chi phí hạ tầng xuống còn **$14,626/tháng**, mang lại mức tiết kiệm **$12,507/tháng (giảm 46.1%)**, đồng thời hạ đơn giá suy luận xuống còn **$1.126 / 1 triệu token (tiết kiệm 82.6% đơn giá token)**.

### Bảng Chỉ Số Cốt Lõi (Baseline vs. Optimized)

| Chỉ số đo lường | Trước tối ưu (Baseline) | Sau tối ưu (Optimized) | Mức cải thiện / Tiết kiệm | Đơn vị tính |
|---|---|---|---|---|
| **Tổng chi phí GPU hàng tháng** | **$27,133** | **$14,626** | **-$12,507 (-46.1%)** | USD / tháng |
| **Đơn giá suy luận trung bình** | **$6.488** | **$1.126** | **-$5.362 (-82.6%)** | $/1M-token |
| **Chi phí mua sắm GPU (Purchasing)** | $25,667 | $15,627 | -$10,040 (-39.1%) | USD / tháng |
| **Lãng phí GPU nhàn rỗi (Idle Waste)** | $600 | $0 | -$600 (-100%) | USD / tháng |
| **Chi phí GPU bị sai lệch cấu hình** | $655 | $0 | -$655 (-100%) | USD / tháng |
| **Tỷ lệ gắn nhãn chi phí (Tag Coverage)** | — | **92%** | **Mở cổng Chargeback** | % |
| **Phát thải carbon tác vụ ngắt quãng** | 679.8 kg | 53.7 kg | **-626.1 kg (-92.1%)** | kgCO2e / batch |

---

## 2. Phân Tích Chi Tiết 4 Đòn Bẩy Tiết Kiệm Chi Phí (FinOps Levers)

```
                       CƠ CẤU TIẾT KIỆM HÀNG THÁNG ($12,507)
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Purchasing Strategy (Spot / Reserved 3yr)         : $10,040 (80.3%)   │
  │ Inference Levers (Cascade / Cache / Batch)        : $1,212   (9.7%)    │
  │ Right-sizing GPU-Util Lies (Decode-bound GPUs)    : $655     (5.2%)    │
  │ Kill Idle GPUs (Utilization < 10%)                : $600     (4.8%)    │
  └────────────────────────────────────────────────────────────────────────┘
```

### 2.1. Đòn bẩy 1: Chiến lược Mua sắm (Purchasing Strategy — Tiết kiệm $10,040/tháng)
- **Cơ chế:** Phân loại 8 tác vụ điện toán theo chu kỳ hoạt động (*duty cycle*) và khả năng chấp nhận gián đoạn (*interruptibility*).
- **Điểm hòa vốn cam kết (*Break-even Utilization*):** Với mức chiết khấu Reserved 3 năm là 45%, điểm hòa vốn là `1 - 0.45 = 55%` (tương đương ≥ 13.2 giờ/ngày).
- **Phân bổ tối ưu:**
  - Các job chạy liên tục 24/7 (`job-infer-chat`, `job-infer-rag`, `job-infer-search`): Chuyển từ On-demand sang **Reserved 3-year** (giảm 40–45% chi phí).
  - Các job huấn luyện có thể ngắt quãng (`job-train-llm`, `job-train-embed`, `job-finetune`, `job-batch-eval`): Chuyển sang **Spot Instance kết hợp Checkpointing định kỳ**. Dù có overhead lưu checkpoint (~3%) và rework khi bị ngắt (~0.5h/lần), Spot vẫn giúp tiết kiệm ~37–40% so với On-demand.

### 2.2. Đòn bẩy 2: Tối ưu hóa Suy luận (Inference Levers — Giảm 82.6% $/1M-token)
Áp dụng công thức chồng chiết khấu (*discount stack*):
$$\text{Effective Cost Fraction} = (\text{CacheHit} \times 0.10 + (1 - \text{CacheHit})) \times (\text{BatchDiscount})$$
1. **Model Cascade:** Định tuyến 65% truy vấn đơn giản sang model nhỏ (giá input/output rẻ hơn 15× so với model lớn).
2. **Prompt Caching:** Chiết khấu 90% cho các đoạn prompt hệ thống đã được lưu cache (chỉ tính 10% giá input).
3. **Batch API:** Gộp các truy vấn phi thời gian thực (như eval, summarization hàng loạt) để hưởng chiết khấu 50%.
- Khi kết hợp 100% cache hit và Batch API, chi phí chỉ còn **5% (0.05)** so với truy vấn ngây thơ ban đầu!

### 2.3. Đòn bẩy 3: Hạ cấp GPU bị "GPU-Util Lie" (Right-sizing — Tiết kiệm $655/tháng)
- Phát hiện các GPU chạy suy luận nghẽn băng thông nhưng thuê GPU đắt tiền (như `gpu-h100-4` chạy H100 giá $2.50/h nhưng MFU chỉ ~20%).
- Chuyển `gpu-h100-4` về `A100` ($1.79/h, tiết kiệm $511/tháng) và `gpu-a10g-1` về `L4` ($0.80/h, tiết kiệm $144/tháng).

### 2.4. Đòn bẩy 4: Tắt GPU nhàn rỗi (Kill Idle GPUs — Tiết kiệm $600/tháng)
- Phát hiện `gpu-h100-5` bị bỏ trống 8 giờ/ngày (utilization < 10%). Tắt hoàn toàn GPU trong các khung giờ nhàn rỗi tiết kiệm ngay **$20/ngày = $600/tháng**.

---

## 3. Phân Tích Chuyên Sâu: Bản Chất "GPU-Util Lie" & Mô Hình Roofline

### 3.1. Tại sao `nvidia-smi` báo 98% GPU-Util lại là một "Lời nói dối"?
Lệnh `nvidia-smi` đo tỷ lệ thời gian mà **GPU engine clock** đang bận xử lý một lệnh nào đó trong chu kỳ lấy mẫu (1 giây). Tuy nhiên:
- GPU-Util **KHÔNG đo lường mức độ sử dụng của các Tensor Core**.
- Khi một kernel đang đợi dữ liệu từ bộ nhớ HBM nạp vào SRAM (hiện tượng **Memory Stall**), hoặc CPU gửi lệnh chậm qua PCIe, GPU clock vẫn được tính là "100% bận", nhưng năng lực tính toán toán học thực tế bằng 0.

```
                      GPU-Util vs. MFU/MBU Reality
┌────────────────────────────────────────────────────────────────────────┐
│  nvidia-smi: GPU-Util = 98.2%  [████████████████████████████░]         │
│  Thực tế FLOPs: MFU   = 19.4%  [█████░░░░░░░░░░░░░░░░░░░░░░░] (Lãng phí)│
│  Thực tế Băng thông:   20.7%  [█████░░░░░░░░░░░░░░░░░░░░░░░]          │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2. Phân tích dưới góc nhìn Roofline Model
- **Pha Prefill:** Xử lý toàn bộ prompt đầu vào song song. Cường độ số học (*Arithmetic Intensity*) rất cao (~455 FLOP/byte) vượt qua điểm đỉnh (*Ridge Point* ~295 FLOP/byte của H100) $\rightarrow$ **Compute-bound** (Khai thác tốt H100).
- **Pha Decode:** Sinh từng token tuần tự theo cơ chế autoregressive. Với batch size nhỏ, cường độ số học chỉ đạt ~1–2 FLOP/byte $\ll 295$ $\rightarrow$ **Memory-bound**.
- **Hệ quả tài chính:** NimbusAI đã thuê GPU H100 ($2.50/h) để chạy decode với batch nhỏ. Do đó, GPU dành 80% thời gian chờ đợi nạp weights từ HBM, khiến doanh nghiệp trả tiền cho "thời gian chờ" của H100 với mức giá đắt đỏ.

---

## 4. Kết Quả Triển Khai Các Phần Mở Rộng "Your Turn" (Phần D)

### 4.1. Extension 1: Cải tiến Chính Sách `recommend_tier()` Đa Yếu Tố
Chúng tôi đã mở rộng hàm `recommend_tier()` trong `finops/pricing.py` để bổ sung 2 yếu tố:
1. **Tỷ lệ gián đoạn thực tế theo kiến trúc GPU (*Interruption Rate Matrix*):** `H100` (4%), `A100` (6%), `A10G` (12%), `L4` (10%). Với các GPU commodity như A10G có tỷ lệ gián đoạn cao >10% trong các job quan trọng kéo dài, thuật toán sẽ tự động tránh rủi ro gián đoạn dây chuyền.
2. **Thời lượng công việc (*Job Duration Risk*):** Với các job ngắn hạn (<30 ngày), thuật toán ưu tiên On-demand hoặc Spot thay vì ký cam kết Reserved 1–3 năm để tránh rủi ro khoá vốn (lock-in).

### 4.2. Extension 2: Right-sizing Dựa trên MBU và $/GB-VRAM
Bảng phân tích danh mục GPU theo chi phí dung lượng bộ nhớ:
- `A100 (80GB)`: $0.0224 / GB-VRAM-hr
- `H100 (80GB)`: $0.0312 / GB-VRAM-hr (Đắt hơn 39% trên mỗi GB VRAM)
- `MI300X (192GB)`: $0.0102 / GB-VRAM-hr (Hiệu quả VRAM cao nhất)

**Kết quả:** Hạ cấp thành công 2 GPU memory-bound, mang lại mức tiết kiệm **$655/tháng** mà không làm suy giảm thời gian đáp ứng (latency SLA).

### 4.3. Extension 3: Kinh Tế Học Prompt Caching & Điểm Hòa Vốn
Triển khai hàm `cache_is_worth_it(avg_cache_reads, write_cost, read_discount)`:
- Với giá ghi cache premium ($3.75/1M token) và giá đọc giảm 90% ($0.30/1M token so với $3.00 gốc), số lần đọc hòa vốn (*Break-even Reads*) là:
$$\text{Break-even Reads} = \frac{\$3.75}{\$3.00 \times (1 - 0.10)} = 1.39 \text{ lần đọc}$$
- Dữ liệu thực tế tại NimbusAI có `avg_cache_reads = 4.2 lần` $\gg 1.39$, khẳng định việc bật prompt caching đem lại lợi nhuận biên ròng rất lớn.

### 4.4. Extension 5: Lập Lịch Nhận Thức Carbon (Carbon-aware Scheduling)
Đánh giá 1,789 kWh điện năng tiêu thụ của các tác vụ huấn luyện ngắt quãng trên 5 vùng điện toán:

| Vùng (Region) | Phát thải lưới điện | Đơn giá điện | Tổng Carbon (kgCO2e) | Chi phí tiền điện | Giảm Carbon so với US-East |
|---|---|---|---|---|---|
| `us-east-1` (Bắc Virginia) | 380 g/kWh | $0.120 / kWh | 679.8 kg | $214.68 | Baseline (0%) |
| `us-west-2` (Oregon Hydro) | 120 g/kWh | $0.070 / kWh | 214.7 kg | $125.23 | **-68.4%** |
| `europe-north1` (Na Uy Thủy điện) | **30 g/kWh** | **$0.090 / kWh** | **53.7 kg** | **$161.01** | **-92.1% (Sạch nhất)** |
| `europe-central2` (Ba Lan Than đá) | 660 g/kWh | $0.180 / kWh | 1,180.7 kg | $322.02 | +73.7% (Ô nhiễm nhất) |
| `us-east-wa` (Washington Hydro) | 90 g/kWh | **$0.055 / kWh** | 161.0 kg | **$98.39** | **-76.3% (Điện rẻ nhất)** |

**Insight:** Di chuyển toàn bộ các job training có thể ngắt quãng sang **`europe-north1`** giúp triệt tiêu **92.1% lượng phát thải carbon** với chi phí điện thấp hơn 25%, mà hoàn toàn không ảnh hưởng đến độ trễ người dùng cuối (do training chạy ngầm phi đồng bộ).

---

## 5. Đánh Giá Tính Bền Vững (Sustainability)

- **Điện năng trung bình mỗi truy vấn (Wh per query):** `0.24 Wh` (truy vấn chuẩn qua model nhỏ).
- **Mức tiêu thụ của truy vấn Reasoning:** `19.2 Wh / query` (**gấp 80 lần** truy vấn chuẩn).
- **Phát thải trung bình:** `0.091 gCO2e / query`.
- **Chính sách khuyến nghị:** Áp dụng cơ chế **Reasoning Gate** — chỉ kích hoạt reasoning tokens khi truy vấn phức tạp (được phân loại bởi model cascade), ngăn chặn lãng phí năng lượng và chi phí bùng nổ.

---

## 6. Top 3 Khuyến Nghị Hành Động Cho FinOps Lead NimbusAI

1. **Hành động 1 (Ưu tiên Cao nhất — ROI tức thì): Triển khai Model Cascade + Prompt Caching mặc định trên API Gateway.**
   - *Lý do:* Giảm ngay 82.6% chi phí token mà không cần thay đổi hạ tầng phần cứng.
2. **Hành động 2 (Ưu tiên Cao — Tiết kiệm tài chính lớn nhất): Ký cam kết Reserved 3 năm cho 3 cụm GPU phục vụ 24/7 và chuyển Training sang Spot + Checkpoint.**
   - *Lý do:* Tiết kiệm ngay $10,040/tháng (chiếm 80.3% tổng mức cắt giảm toàn công ty).
3. **Hành động 3 (Ưu tiên Trung hạn — Quản trị & Bền vững): Thiết lập Chargeback bắt buộc theo chuẩn FOCUS và tự động định tuyến Training sang `europe-north1`.**
   - *Lý do:* Với Tag Coverage đã đạt 92% (vượt ngưỡng 80%), việc chargeback sẽ phân định trách nhiệm ngân sách cho từng team, đồng thời giảm 92% lượng carbon phát thải của công ty.

---

## Phụ Lục: Giải Đáp 5 Câu Hỏi Kiểm Tra Hiểu Biết (Oral Check)

1. **GPU-Util 98% có nghĩa là GPU đang làm việc hiệu quả không? Tại sao?**
   - *Trả lời:* Không. GPU-Util chỉ đo thời gian xung nhịp bận, không đo hiệu quả tính toán Tensor Core. Khi GPU bị nghẽn nạp bộ nhớ (memory stall) hoặc truyền dữ liệu PCIe, GPU-Util vẫn báo 98% nhưng MFU thực tế có thể dưới 20%.
2. **Tại sao cần ≥ 80% tag coverage mới dám thực hiện Chargeback?**
   - *Trả lời:* Nếu tag coverage < 80%, trên 20% chi phí không xác định được chủ sở hữu. Việc thu tiền (chargeback) khi dữ liệu thiếu hụt sẽ dẫn đến tranh chấp giữa các phòng ban và mất niềm tin vào hệ thống FinOps. Thay vào đó, chỉ nên dùng Showback (thông báo nhận thức) cho đến khi đạt ≥ 80%.
3. **Nếu công ty có 70% workload interruptible, bạn sẽ tối ưu purchasing như thế nào?**
   - *Trả lời:* Chuyển toàn bộ 70% workload này sang Spot Instance kết hợp cơ chế lưu Checkpoint tự động định kỳ lên Object Storage. Điều này giúp hưởng mức chiết khấu spot 40–60% với chi phí bù đắp rủi ro gián đoạn chỉ ~3–5%.
4. **Đo bằng $/GPU-hr vs $/1M-token — khi nào con số này cho kết quả trái ngược nhau?**
   - *Trả lời:* Khi tối ưu phần mềm (như bật TensorRT-LLM, vLLM, continuous batching, FP8 quantization). Một GPU H100 có giá `$2.50/GPU-hr` (đắt gấp 2.5× so với A10G `$1.00/GPU-hr`), nhưng H100 có thể phục vụ số token gấp 10× A10G. Khi đó, `$/GPU-hr` tăng nhưng `$/1M-token` lại giảm mạnh 75%.
5. **Tại sao LLM decode là memory-bound còn prefill là compute-bound?**
   - *Trả lời:* Pha Prefill nạp toàn bộ ma trận trọng số một lần và tính toán đồng thời trên $N$ tokens đầu vào ($N$ lớn $\rightarrow$ Arithmetic Intensity cao $\rightarrow$ Compute-bound). Pha Decode phải nạp lại toàn bộ ma trận trọng số khổng lồ chỉ để sinh ra duy nhất 1 token tiếp theo ($N=1 \rightarrow$ Arithmetic Intensity cực thấp ~1 FLOP/byte $\rightarrow$ GPU bị giới hạn hoàn toàn bởi băng thông bộ nhớ HBM).
