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
#include "stencil-composition/timer.hpp"

namespace gridtools {

    /**
    * @class timer_cuda
    * CUDA implementation of the Timer interface
    */
    class timer_cuda : public timer< timer_cuda > // CRTP
    {
      public:
        __host__ timer_cuda(std::string name) : timer< timer_cuda >(name) {
            // create the CUDA events
            cudaEventCreate(&start_);
            cudaEventCreate(&stop_);
        }
        __host__ ~timer_cuda() {
            // free the CUDA events
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }

        /**
        * Reset counters
        */
        __host__ void set_impl(double const & /*time_*/) {}

        /**
        * Start the stop watch
        */
        __host__ void start_impl() {
            // insert a start event
            cudaEventRecord(start_, 0);
        }

        /**
        * Pause the stop watch
        */
        __host__ double pause_impl() {
            // insert stop event and wait for it
            cudaEventRecord(stop_, 0);
            cudaEventSynchronize(stop_);

            // compute the timing
            float result;
            cudaEventElapsedTime(&result, start_, stop_);
            return static_cast< double >(result) * 0.001f; // convert ms to s
        }

      private:
        cudaEvent_t start_;
        cudaEvent_t stop_;
    };
}
