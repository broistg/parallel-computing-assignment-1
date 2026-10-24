#include "barrier.h"
#include <thread>
#include <chrono>
#include <vector>
#include <iostream>

using namespace std::chrono;

/* Configurations */
#define NUM_OF_THREADS 16
#define NUM_OF_ITERS 100000
#define N 10007

/* Shared variables */
Barrier barrier(NUM_OF_THREADS);
std::vector<int> shared_data(NUM_OF_THREADS, 0);
std::atomic<bool> error_detected{false};


/* Child thread's code */
void thread_entry(int tid) {
    for (auto i = 0; i < NUM_OF_ITERS; ++i) {
        // create unique value
        shared_data[tid] = i + 1;
        barrier.await();

        // check Invariant
        for (int j = 0; i < NUM_OF_THREADS; ++i) {
            if (shared_data[j] != i + 1) {
                error_detected.store(true);
            }
        }
        barrier.await();
    }
}

int main(int argc, char* argv[]) {
    std::vector<std::thread> threads;
    threads.reserve(NUM_OF_THREADS);
    auto start = steady_clock::now();

    // create threads
    for (int i = 0; i < NUM_OF_THREADS; ++i) {
        threads.emplace_back(thread_entry, i);
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    auto end = steady_clock::now();
    if(!error_detected.load()) {
        std::cout << "Kiểm thử thất bại: Vi phạm Invariant của Barrier!\n";
    } else {
        std::cout << "[PASS] Tính Correctness được kiểm chứng với " << NUM_OF_THREADS 
              << " luồng với " << NUM_OF_ITERS << " vòng lặp.\n";
    }
    std::cout<<"Thời gian chạy: "<<duration_cast<milliseconds>(end - start).count();
    return 0;
}