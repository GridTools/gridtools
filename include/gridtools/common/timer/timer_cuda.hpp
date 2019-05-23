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
            return {event};
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
