#pragma once

#include <stencil-composition/timer.hpp>

namespace gridtools{

/**
* @class TimerDummy
* Dummy timer implementation doing nothing in order to avoid runtime overhead
*/
class timer_dummy : public timer<timer_dummy> // CRTP
{
public:
    timer_dummy(std::string name) : timer(name) {}
    ~timer_dummy() {}

    /**
    * Reset counters
    */
    void reset_impl() {}

    /**
    * Start the stop watch
    */
    void start_impl() {}

    /**
    * Pause the stop watch
    */
    double pause_impl() { return 0.0; }
};
}
