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

#if defined(GT_USE_GPU) && defined(GT_ENABLE_METERS)

#include <type_traits>

namespace gridtools {
    /**
     * @class timer_cuda
     */
    class timer_cuda {
        struct destroy_event {
            inline void operator()(cudaEvent_t *ptr) {
                cudaEventDestroy(*ptr);
                delete ptr;
            }
        };

        using event_holder = std::unique_ptr<cudaEvent_t, destroy_event>;

        static event_holder create_event() {
            auto event = std::make_unique<cudaEvent_t> event;
            GT_CUDA_CHECK(cudaEventCreate(event.get()));
            return event_holder{event.relese()};
        }

        event_holder m_start = create_event();
        event_holder m_stop = create_event();

      public:
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

#else

#include "timer_dummy.hpp"

namespace gridtools {
    using timer_cuda = timer_dummy;
}

#endif
