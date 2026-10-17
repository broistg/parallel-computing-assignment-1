#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <iomanip>

using namespace std::chrono;

struct Measurement {
    int step;
    unsigned param_val;
    double actual_time_us;
};

void test_sleep_backoff(int steps, unsigned start_ns, unsigned max_ns) {
    std::cout << "=== THỬ NGHIỆM 1: std::this_thread::sleep_for(nanoseconds) ===\n";
    std::cout << std::setw(6) << "Step" 
              << std::setw(18) << "Requested (ns)" 
              << std::setw(22) << "Actual Time (us)" 
              << std::setw(15) << "Factor x2?\n";
    std::cout << std::string(62, '-') << "\n";

    unsigned delay_ns = start_ns;
    double prev_time = 0.0;

    for (int i = 1; i <= steps; ++i) {
        auto t_start = high_resolution_clock::now();
        std::this_thread::sleep_for(nanoseconds(delay_ns));
        auto t_end = high_resolution_clock::now();

        double actual_us = duration<double, std::micro>(t_end - t_start).count();
        
        std::cout << std::setw(6) << i
                  << std::setw(18) << delay_ns
                  << std::setw(22) << std::fixed << std::setprecision(2) << actual_us;

        if (i > 1 && prev_time > 0.0) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << (actual_us / prev_time) << "x\n";
        } else {
            std::cout << std::setw(15) << "-\n";
        }

        prev_time = actual_us;
        if (delay_ns < max_ns) {
            delay_ns *= 2;
        }
    }
    std::cout << "\n";
}

void test_yield_backoff(int steps, unsigned start_yields, unsigned max_yields) {
    std::cout << "=== THỬ NGHIỆM 2: Vòng lặp std::this_thread::yield() ===\n";
    std::cout << std::setw(6) << "Step" 
              << std::setw(18) << "Yield Count" 
              << std::setw(22) << "Actual Time (us)" 
              << std::setw(15) << "Factor x2?\n";
    std::cout << std::string(62, '-') << "\n";

    unsigned yield_count = start_yields;
    double prev_time = 0.0;

    for (int i = 1; i <= steps; ++i) {
        auto t_start = high_resolution_clock::now();
        
        // Vòng lặp nhường quyền CPU theo số lượt nhân đôi
        for (unsigned y = 0; y < yield_count; ++y) {
            std::this_thread::yield();
        }
        
        auto t_end = high_resolution_clock::now();

        double actual_us = duration<double, std::micro>(t_end - t_start).count();

        std::cout << std::setw(6) << i
                  << std::setw(18) << yield_count
                  << std::setw(22) << std::fixed << std::setprecision(2) << actual_us;

        if (i > 1 && prev_time > 0.0) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << (actual_us / prev_time) << "x\n";
        } else {
            std::cout << std::setw(15) << "-\n";
        }

        prev_time = actual_us;
        if (yield_count < max_yields) {
            yield_count *= 2;
        }
    }
    std::cout << "\n";
}

int main() {
    const int NUM_STEPS = 14;

    // Đoạn chạy Sleep: từ 100 ns đến tối đa ~1.000.000 ns (1 ms)
    test_sleep_backoff(NUM_STEPS, 100, 1000000);

    // Đoạn chạy Yield loop: từ 1 lượt yield đến tối đa 8192 lượt
    test_yield_backoff(NUM_STEPS, 1, 8192);

    return 0;
}