#include "barrier.h"
#include <atomic>
#include <thread>

#define CPU_PAUSE() std::this_thread::yield()

Barrier::Barrier(int n) {
    count = n;
    size = n;
    globalSense = false;
}

void Barrier::await() {
    thread_local bool mySense = !globalSense;
    int position = count.fetch_sub(1);

    if (position == 1) {
        count.store(size);
        globalSense = mySense;
    } else {
        int backoff_limit = 1;
        const int MAX_BACKOFF = 1024;
        while (globalSense != mySense) {
            for (int i = 0; i < backoff_limit; ++i) {
                CPU_PAUSE();
            }
            if (backoff_limit < MAX_BACKOFF) {
                backoff_limit *= 2;
            }
        }
    }
    mySense = !globalSense;
}