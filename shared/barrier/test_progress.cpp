#include "barrier.h"
#include <thread>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>

using namespace std::chrono;

/* Configurations */
#define NUM_OF_THREADS 16
#define NUM_OF_ITERS 10000
#define N 10007
#define TIMEOUT_THRESHOLD milliseconds(10000)

/* Shared variables */
Barrier barrier(NUM_OF_THREADS);
std::atomic<int> completed_episodes{0};
std::atomic<bool> test_finished{false};

/* Child thread's code */
void thread_entry(int tid) {
    // create delay in [0 - 50] microsec
    std::mt19937 rng(1337 + tid);
    std::uniform_int_distribution<int> jitter_dist(0, 50);
    
    for (auto i = 0; i < NUM_OF_ITERS; ++i) {
        // random delay
        if (jitter_dist(rng) > 40) {
            std::this_thread::sleep_for(std::chrono::microseconds(jitter_dist(rng)));
        }
        barrier.await();

        if (tid == 0) {
            completed_episodes.fetch_add(1, std::memory_order_relaxed);
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

    // create Watchdog Thread check liveness
    int last_observed_episode = 0;
    auto last_progress_time = steady_clock::now();

    while (!test_finished.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        int current_ep = completed_episodes.load(std::memory_order_relaxed);
        
        // check stop condition
        if (current_ep >= NUM_OF_ITERS) {
            test_finished.store(true);
            break;
        }

        auto now = std::chrono::steady_clock::now();
        if (current_ep > last_observed_episode) {
            // Do progress: reset timer
            last_observed_episode = current_ep;
            last_progress_time = now;
        } else {
            // Don't progress: check if it is timeout
            if (now - last_progress_time > TIMEOUT_THRESHOLD) {
                std::cerr << "\n[VIOLATION] Progress Property Failed!\n"
                          << "Deadlock/Livelock phát hiện tại episode: " << last_observed_episode 
                          << "\n";
                std::terminate();
            }
        }
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    auto end = steady_clock::now();
    std::cout << "[PASS] Tính Progress được kiểm chứng với " << NUM_OF_THREADS 
            << " luồng với " << NUM_OF_ITERS << " vòng lặp.\n";
    std::cout<<"Thời gian chạy: "<<duration_cast<milliseconds>(end - start).count();
    return 0;
}