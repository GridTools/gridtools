#pragma once
#include <string>
#include "stencil_composition/timer.hpp"

namespace gridtools {

    /**
    * @class TimerDummy
    * Dummy timer implementation doing nothing in order to avoid runtime overhead
    */
    class timer_dummy : public timer< timer_dummy > // CRTP
    {
      public:
        __host__ timer_dummy(std::string name) : timer< timer_dummy >(name) {}
        __host__ ~timer_dummy() {}

        /**
        * Reset counters
        */
        __host__ void reset_impl() {}

        /**
        * Start the stop watch
        */
        __host__ void start_impl() {}

        /**
        * Pause the stop watch
        */
        __host__ double pause_impl() { return 0.0; }
    };
}
