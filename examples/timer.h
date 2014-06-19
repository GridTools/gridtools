#pragma once
#include <chrono>

class timer
{
private:
    std::chrono::high_resolution_clock::time_point start_;
public:
    timer()
    {
        reset();
    }
    void reset()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }
    std::chrono::milliseconds elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_);
    }
    friend std::ostream &operator<<(std::ostream &sout, timer const &t)
    {
        return sout << t.elapsed().count() << "ms";
    }
};
