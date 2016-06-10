#pragma once

#include "stencil_composition/timer.hpp"
#include "common/defs.hpp"

namespace gridtools {

    /**
    * @class timer_host
    * host implementation of the Timer interface
    */
    class timer_host : public timer< timer_host > // CRTP
    {
      public:
        timer_host(std::string name) : timer< timer_host >(name) { startTime_ = 0.0; }
        ~timer_host() {}

        /**
        * Reset counters
        */
        void reset_impl() { startTime_ = 0.0; }

        /**
        * Start the stop watch
        */
        void start_impl() {
#if defined(_OPENMP)
            startTime_ = omp_get_wtime();
#endif
        }

        /**
        * Pause the stop watch
        */
        double pause_impl() {
#if defined(_OPENMP)
            return omp_get_wtime() - startTime_;
#else
            return -100;
#endif
        }

      private:
        double startTime_;
    };
}
