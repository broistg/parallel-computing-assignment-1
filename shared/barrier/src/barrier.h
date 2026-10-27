#include <atomic>
#include <thread>

class Barrier {
private:
    std::atomic<int> count;
    int size;
    bool globalSense;
public:
    explicit Barrier(int n);
    ~Barrier() = default;
    void await();
};