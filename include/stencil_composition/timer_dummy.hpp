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
