/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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
