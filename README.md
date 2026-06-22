# WORKFLOW PROJECT: HỆ THỐNG DISTRIBUTED MULTI-HEAD ATTENTION (HYBRID PARALLELISM)

Tài liệu này xác định luồng công việc tiêu chuẩn cho nhóm 4 kỹ sư, thiết kế và phát triển một hệ thống AI cốt lõi từ con số 0 bằng C/C++ thuần và OpenMPI. Kiến trúc mục tiêu là **Hybrid Parallelism**, kết hợp phân tán theo cụm (Head Parallelism) và phân tán theo ma trận (Tensor Parallelism).



---

## 1. Giai đoạn 1: Thiết lập Hạ tầng (VLAN / Remote PoC)

Mục tiêu: Xây dựng môi trường phát triển phân tán cho phép làm việc từ xa, đảm bảo mã nguồn biên dịch và chạy chính xác về mặt logic toán học trước khi tối ưu hóa tốc độ.

* **Cụm 4 node:** Hệ thống chạy trên **4 máy** theo quy ước đặt tên `master`, `slave1`, `slave2`, `slave3` (khớp với bộ SSH key `id_rsa_master`, `id_rsa_slave1`, ...).
* **Hạ tầng mạng ảo (VLAN):** Sử dụng Tailscale hoặc ZeroTier cài đặt trực tiếp bên trong 4 máy ảo Ubuntu (không dùng Radmin VPN trên Windows Host). Mỗi node nhận một IP tĩnh ảo (VD: `10.147.20.x`).
* **Đồng bộ hóa môi trường (Dependency Freeze):** Thống nhất 100% phiên bản hệ điều hành (Ubuntu 22.04/24.04 LTS), trình biên dịch GCC và **cùng một phiên bản OpenMPI trên cả 4 máy**. Tuyệt đối không tự ý nâng cấp thư viện cá nhân.
* **Phân giải tên miền & Xác thực:** Cập nhật file `/etc/hosts` ánh xạ `master`/`slave1`/`slave2`/`slave3` sang IP, và thiết lập Passwordless SSH bằng RSA Key từ `master` sang tất cả các `slaveN`.
* **Quản lý mã nguồn (NFS):** Dùng **NFS** (server ở `master`, client ở các `slaveN`) để chia sẻ thư mục dự án. Biên dịch **một lần** trên `master` trong thư mục NFS → binary tự động xuất hiện trên tất cả các node (thay cho việc `git pull` về cùng đường dẫn trên từng máy). Dữ liệu đầu vào được sinh trực tiếp trong bộ nhớ nên không cần chia sẻ file dữ liệu.

### Quy trình khởi động cụm (trên `master`)

1. Đặt `slots=` trong file `hostfile` bằng số nhân (`nproc`) của từng máy.
2. **Kiểm tra cụm trước (Part 4):** `bash scripts/test_cluster.sh` — biên dịch và chạy `mpi-prime` trên toàn bộ node. Chỉ tiếp tục khi báo `cluster OK`.
3. **Biên dịch dự án:** `make` (tạo cả `hybrid_attention` lẫn `mpi-prime`).
4. **Chạy:** `mpirun --hostfile hostfile -np <tổng_số_nhân> ./hybrid_attention --mode hybrid --seq-len 1024`

---

## 2. Yêu cầu Hệ thống và Kiến trúc Mã nguồn (Requirements)

Hệ thống yêu cầu triển khai phân tán phương trình cốt lõi của Transformer:
$$Attention(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V$$

### A. Phân rã Module Hệ thống (> 1000 LOC)

| Module | Chức năng cốt lõi | Yêu cầu Kỹ thuật Kịch trần |
| :--- | :--- | :--- |
| **Tensor & Memory Core** | Quản lý vòng đời dữ liệu | Cấp phát/giải phóng động an toàn. Triển khai toán tuyến tính: Dot Product, Matrix Transpose. |
| **MPI Topology Manager** | Quản trị không gian giao tiếp | Triển khai `MPI_Comm_split` để tạo các sub-communicators. Cấu hình lưới Cartesian 2x2. |
| **Hybrid Attention Logic** | Thuật toán AI phân tán | Tích hợp thuật toán Cannon (nhân khối). Xây dựng Distributed Softmax chống lỗi tràn số (Overflow). |
| **I/O & Profiling** | Nạp dữ liệu và kiểm chứng | Đọc tensor từ file `.bin`. Đo lường bằng `MPI_Wtime()` tách biệt pha I/O, Tính toán và Mạng. |

---

## 3. Lộ trình Phát triển Phân lớp (Phased Approach)

Để quản trị rủi ro Deadlock và Segmentation Fault, dự án bắt buộc tuân thủ 3 cột mốc (Milestones) tích hợp tăng dần:

### Milestone 1 (Tuần 1): Head Parallelism Thuần túy
* **Chiến lược:** Mỗi node/nhóm node phụ trách trọn vẹn một hoặc nhiều Attention Head.
* **Mục tiêu kỹ thuật:** Chạy thành công `MPI_Scatter` chia dữ liệu Head và `MPI_Gather` để gom kết quả. Nhân ma trận diễn ra cục bộ trên từng node.

### Milestone 2 (Tuần 2): Đột phá Tensor Parallelism
* **Chiến lược:** Coi hệ thống chỉ có 1 Head duy nhất. Ép 4 node hợp tác để giải một phép nhân ma trận cực lớn và tính toán Softmax phân tán.
* **Mục tiêu kỹ thuật:** Triển khai cơ chế luân chuyển khối ma trận (Block shifting) bằng `MPI_Sendrecv_replace`. Hoàn thiện hàm Softmax an toàn tích hợp `MPI_Allreduce`.

### Milestone 3 (Tuần 3): Dung hợp Hybrid Integration
* **Chiến lược:** Tích hợp M2 vào M1. Chia 4 node thành 2 nhóm nhỏ.
* **Mục tiêu kỹ thuật:** Gọi thành công `MPI_Comm_split`. Đảm bảo luồng tin nhắn (message passing) của việc nhân khối ma trận bị cô lập nghiêm ngặt bên trong từng sub-communicator.

---

## 4. Giai đoạn 4: Chuyển đổi Local LAN & Benchmarking

Di chuyển từ mạng ảo (VLAN) sang mạng nội bộ vật lý (Gigabit Switch / LAN cục bộ) để đo lường hiệu năng thực tế.

### A. Quy trình Migration
1. Tắt dịch vụ VLAN (Tailscale/ZeroTier).
2. Kết nối các máy host Windows vào chung mạng LAN. Cấu hình máy ảo Ubuntu nhận Bridged Adapter.
3. Cập nhật IP mới của dải mạng vật lý vào `/etc/hosts` trên cả 4 node.
4. Cập nhật Interface Flag: `mpirun -np 4 --hostfile mpi_hosts --mca btl_tcp_if_include <tên_card_Bridged> ./hybrid_attention`

### B. Tiêu chuẩn Đánh giá
* **Đo lường thời gian (Execution Time):** Tách biệt thời gian I/O và Computation.
* **Speedup Factor:** Tính hệ số tăng tốc với ma trận lớn ($N \ge 4096$).
* **Network Overhead Profiling:** Đánh giá giới hạn băng thông khi luân chuyển Tensor khối giữa các node.

---

## 5. Cảnh báo Rủi ro Nghiêm trọng

1. **Ảo giác Toán học Phân tán (Silent Mathematical Hallucination):** Ở Milestone 3, nếu chỉ định sai `comm` (ví dụ dùng `MPI_COMM_WORLD` thay vì `sub_comm` trong `MPI_Allreduce`), hệ thống sẽ không báo lỗi biên dịch, code vẫn chạy ra số nhưng ma trận trọng số bị hòa trộn sai lệch hoàn toàn. Việc đối chiếu PyTorch Baseline sau mỗi commit là **Bắt Buộc**.
2. **Nút thắt VLAN (Bottleneck):** Ở Tuần 1 và 2, khi chạy test trên Tailscale, code có thể chạy chậm hơn cả chạy tuần tự trên 1 máy do overhead mạng ảo. Đây là hiện tượng vật lý bình thường, nghiêm cấm việc cố gắng tối ưu hóa (premature optimization) trong giai đoạn này. Hiệu năng thực sự chỉ bộc lộ ở mạng vật lý (Giai đoạn 4).