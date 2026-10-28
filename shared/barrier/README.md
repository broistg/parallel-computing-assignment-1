# Barrier Synchronization in C++

## Mô tả dự án

Dự án hiện thực và kiểm thử cơ chế đồng bộ rào cản (**Barrier Synchronization**) đa luồng trong C++. Project bao gồm các bài kiểm tra
- tính đúng đắn (correctness)
- tính tiến trình (progress)
- đánh giá hiệu năng (benchmark)

## Cấu trúc thư mục
.
├── Dockerfile              # Dockerfile để build và chạy trong môi trường đồng nhất
├── barrier.cpp             # Chứa mã nguồn của giải thuật reversing-sense barrier
├── test_correctness.cpp    # Kiểm tra tính đúng đắn của Barrier (logic chặn, reset chu kỳ)
├── test_progress.cpp       # Kiểm tra tránh deadlock / đảm bảo luồng tiếp tục tiến trình
├── benchmark.cpp                # Đo lường hiệu năng và độ trễ dưới tải luồng cao
└── README.md

## Hướng dẫn chạy

Bước 1: Chọn bài kiểm tra muốn thử (VD: Kiểm tra tính đúng đắn) mặc định là bài kiểm tra performance và scability
Bước 2: Đổi tên file của bài kiểm tra tương ứng thành main.cpp (VD: test_correctness.cpp -> main.cpp)
### Với docker
Bước 3: Chạy câu lệnh "docker build -t bt2:0.0.1 ." để xây dựng image trên docker
Bước 4: Chạy câu lệnh "docker run --rm bt2:0.0.1 ." để chạy mã
Bước 5: Thay đổi các thống số như NUM_OF_THREADS, NUM_OF_ITERS, N để thử nghiệm các tiến trình khác nhau
### Với C++ Compiler
Bước 3: Chạy câu lệnh "g++ -fopenmp -std=c++20 -pthread -o myapp main.cpp barrier.cpp"

## Yêu cầu hệ thống
Docker hoặc C++ Compiler (C++ >= 20)
Hỗ trợ POSIX Threads (-pthread)
Hỗ trợ OpenPM (fopenmp)