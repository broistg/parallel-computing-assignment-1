#include "barrier.h"
#include <thread>
#include <chrono>
#include <vector>
#include <iostream>
#include <barrier>
#include <pthread.h>
#include <functional>
#include <omp.h>
#include <memory>

using namespace std::chrono;

/* Configurations */
int NUM_OF_THREADS = 2;
#define NUM_OF_ITERS 10000
#define N 10007

/* Shared variables */
std::unique_ptr<Barrier> sense_barrier;
std::unique_ptr<std::barrier<>> std_barrier;
pthread_barrier_t pthread_barrier;

/* Private variables */
// thread_Local int dummy;

int is_prime(int p) {
	int d;
	for (d = 2; d < p; d = d +1) {
		if (p % d == 0) return 0;
	}
	return 1;
}

/* Child thread's code */
void thread_sense_barrier_entry() {
	for (auto i = 0; i < NUM_OF_ITERS; ++i) {
		// check if a number N is a prime
		is_prime(N);
		sense_barrier->await();
	}
}

void thread_std_barrier_entry() {
	for (auto i = 0; i < NUM_OF_ITERS; ++i) {
		// check if a number N is a prime
		is_prime(N);
		std_barrier->arrive_and_wait();
	}
}

void thread_pthread_barrier_entry() {
	for (auto i = 0; i < NUM_OF_ITERS; ++i) {
		// check if a number N is a prime
		is_prime(N);
		pthread_barrier_wait(&pthread_barrier);
	}
}

/* Performance method */
double bench_barrier(std::function<void()> thread_entry) {
	pthread_barrier_init(&pthread_barrier, nullptr, NUM_OF_THREADS);
	std::vector<std::thread> threads;
	threads.reserve(NUM_OF_THREADS);

	auto start = steady_clock::now();

	for (int i = 0; i < NUM_OF_THREADS; ++i) {
		threads.emplace_back(thread_entry);
	}

	for (auto& t : threads) {
		if (t.joinable()) t.join();
	}

	auto end = steady_clock::now();
	double sec = duration<double>(end - start).count();
	pthread_barrier_destroy(&pthread_barrier);
	return sec;
}

double bench_OpenPM_barrier() {
    omp_set_num_threads(NUM_OF_THREADS);

    auto start = steady_clock::now();

    #pragma omp parallel
    {
        for (auto i = 0; i < NUM_OF_ITERS; ++i) {
            // check if a number N is a prime
            is_prime(N);
            #pragma omp barrier
        }
    }

    auto end = steady_clock::now();
    double sec = duration<double>(end - start).count();
    pthread_barrier_destroy(&pthread_barrier);
    return sec;
}

int main(int argc, char* argv[]) {
    for (int i = 2; i <=16; i+=2) {
        NUM_OF_THREADS = i;
        sense_barrier = std::make_unique<Barrier>(NUM_OF_THREADS);
        std_barrier = std::make_unique<std::barrier<>>(NUM_OF_THREADS);

        std::cout<< "Running with " << NUM_OF_THREADS << " threads and " << NUM_OF_ITERS << " interations\n";
        std::cout<< "Sense barrier elasp time: " << bench_barrier(thread_sense_barrier_entry) << " sec\n";
        std::cout<< "C++ 20 barrier elasp time: " << bench_barrier(thread_std_barrier_entry) << " sec\n";
        std::cout<< "Pthread libary barrier elasp time: " << bench_barrier(thread_pthread_barrier_entry) << " sec\n";
        std::cout<< "OpenMP barrier elasp time: " << bench_OpenPM_barrier() << " sec\n";
    }
	return 0;
}