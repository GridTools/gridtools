/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include "../host_device.hpp"
#include <cmath>
#include <sstream>
#include <string>
#include <utility>

namespace gridtools {

    /**
     * @class Timer
     * Measures total elapsed time between all start and stop calls
     */
    template <typename TimerImpl>
    class timer {
      protected:
        timer(std::string name) : m_name(std::move(name)) {}

      public:
        /**
         * Reset counters
         */
        GT_FUNCTION_HOST void reset() {
            m_total_time = 0;
            m_counter = 0;
        }

        /**
         * Start the stop watch
         */
        GT_FUNCTION_HOST void start() { impl().start_impl(); }

        /**
         * Pause the stop watch
         */
        GT_FUNCTION_HOST void pause() {
            m_total_time += impl().pause_impl();
            m_counter++;
        }

        /**
         * @return total elapsed time [s]
         */
        GT_FUNCTION_HOST double total_time() const { return m_total_time; }

        /**
         * @return how often the timer was paused
         */
        GT_FUNCTION_HOST size_t count() const { return m_counter; }

        /**
         * @return total elapsed time [s] as string
         */
        GT_FUNCTION_HOST std::string to_string() const {
            std::ostringstream out;
            if (m_total_time < 0 || std::isnan(m_total_time))
                out << m_name << "\t[s]\t"
                    << "NO_TIMES_AVAILABLE"
                    << " (" << m_counter << "x called)";
            else
                out << m_name << "\t[s]\t" << m_total_time << " (" << m_counter << "x called)";
            return out.str();
        }

      private:
        TimerImpl &impl() { return *static_cast<TimerImpl *>(this); }

        std::string m_name;
        double m_total_time = 0;
        size_t m_counter = 0;
    };
} // namespace gridtools
