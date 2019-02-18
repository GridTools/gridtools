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

#include "../cuda_util.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <string>

#include "timer.hpp"

namespace gridtools {

    /**
     * @class timer_cuda
     */
    class timer_cuda : public timer<timer_cuda> // CRTP
    {
        using event_holder =
            std::unique_ptr<CUevent_st, std::integral_constant<decltype(&cudaEventDestroy), cudaEventDestroy>>;

        static event_holder create_event() {
            cudaEvent_t event;
            GT_CUDA_CHECK(cudaEventCreate(&event));
            return event_holder{event};
        }

        event_holder m_start = create_event();
        event_holder m_stop = create_event();

      public:
        timer_cuda(std::string name) : timer<timer_cuda>(name) {}

        void set_impl(double) {}

        void start_impl() {
            // insert a start event
            GT_CUDA_CHECK(cudaEventRecord(m_start.get(), 0));
        }

        double pause_impl() {
            // insert stop event and wait for it
            GT_CUDA_CHECK(cudaEventRecord(m_stop.get(), 0));
            GT_CUDA_CHECK(cudaEventSynchronize(m_stop.get()));

            // compute the timing
            float result;
            GT_CUDA_CHECK(cudaEventElapsedTime(&result, m_start.get(), m_stop.get()));
            return result * 0.001; // convert ms to s
        }
    };
} // namespace gridtools
