/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../cuda_util.hpp"
#include "../hip_wrappers.hpp"
#include <memory>
#include <string>

#include "timer.hpp"

namespace gridtools {

    /**
     * @class timer_cuda
     */
    class timer_cuda : public timer<timer_cuda> // CRTP
    {
        struct destroy_event {
            inline void operator()(cudaEvent_t* ptr) { cudaEventDestroy(*ptr); }
        };

        using event_holder =
            std::unique_ptr<cudaEvent_t, destroy_event>;

        static event_holder create_event() {
            cudaEvent_t* event = new cudaEvent_t;
            GT_CUDA_CHECK(cudaEventCreate(event));
            return event_holder{event};
        }

        event_holder m_start = create_event();
        event_holder m_stop = create_event();

      public:
        timer_cuda(std::string name) : timer<timer_cuda>(name) {}

        void set_impl(double) {}

        void start_impl() {
            // insert a start event
            GT_CUDA_CHECK(cudaEventRecord(*m_start, 0));
        }

        double pause_impl() {
            // insert stop event and wait for it
            GT_CUDA_CHECK(cudaEventRecord(*m_stop, 0));
            GT_CUDA_CHECK(cudaEventSynchronize(*m_stop));

            // compute the timing
            float result;
            GT_CUDA_CHECK(cudaEventElapsedTime(&result, *m_start, *m_stop));
            return result * 0.001; // convert ms to s
        }
    };
} // namespace gridtools
