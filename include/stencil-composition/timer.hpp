/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include <sstream>
#include <string>

namespace gridtools {

    /**
    * @class Timer
    * Measures total elapsed time between all start and stop calls
    */
    template < typename TimerImpl >
    class timer {
        DISALLOW_COPY_AND_ASSIGN(timer);

      protected:
        __host__ timer(std::string name) {
            m_name = name;
            reset();
        }
        __host__ ~timer() {}

      public:
        /**
        * Reset counters
        */
        __host__ void reset() {
            m_total_time = 0.0;
            static_cast< TimerImpl * >(this)->reset_impl();
        }

        /**
        * Start the stop watch
        */
        __host__ void start() { static_cast< TimerImpl * >(this)->start_impl(); }

        /**
        * Pause the stop watch
        */
        __host__ void pause() { m_total_time += static_cast< TimerImpl * >(this)->pause_impl(); }

        /**
        * @return total elapsed time [s]
        */
        __host__ double total_time() const { return m_total_time; }

        /**
        * @return total elapsed time [s] as string
        */
        __host__ std::string to_string() const {
            std::ostringstream out;
            if (m_total_time < 0)
                out << "\t[s]\t" << m_name << "NO_TIMES_AVAILABLE";
            else
                out << m_name << "\t[s]\t" << m_total_time;
            return out.str();
        }

      private:
        std::string m_name;
        double m_total_time;
    };
}
