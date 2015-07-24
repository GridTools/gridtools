#pragma once

#include <stencil-composition/timer.hpp>

#ifdef __ENABLE_OPENMP__
#include <omp.h>
#else
#include <ctime>
#endif

namespace gridtools {

/**
* @class timer_host
* host implementation of the Timer interface
*/
class timer_host : public timer<timer_host> // CRTP
{
public:
    timer_host(std::string name) : timer(name) { startTime_ = 0.0; }
    ~timer_host() {}

    /**
    * Reset counters
    */
    void reset_impl()
    {
        startTime_ = 0.0;
    }

    /**
    * Start the stop watch
    */
    void start_impl()
    {
        startTime_ = omp_get_wtime();
    }

    /**
    * Pause the stop watch
    */
    double pause_impl()
    {
        return omp_get_wtime() - startTime_;
    }

private:
    double startTime_;
};
}
