/*
 * GridTools
 *
 * Copyright (c) 2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <hpx/parallel/algorithm.hpp>

namespace gridtools {
    namespace thread_pool {
        struct hpx {

            friend auto thread_pool_get_thread_num(hpx) {
                return ::hpx::get_worker_thread_num();
            }
            friend auto thread_pool_get_max_threads(hpx) {
                return ::hpx::get_num_worker_threads();
            }

            template <class F, class I>
            friend void thread_pool_parallel_for_loop(hpx, F const &f, I lim) {
                using ::hpx::parallel::for_loop;
                using ::hpx::parallel::execution::par;
                for_loop(par, 0, lim, f);
            }

        };
    } // namespace thread_pool
} // namespace gridtools
