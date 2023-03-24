/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

namespace gridtools {
    namespace thread_pool {
        struct dummy {
            friend auto thread_pool_get_thread_num(dummy) { return 0; }
            friend auto thread_pool_get_max_threads(dummy) { return 1; }

            template <class F, class I>
            friend void thread_pool_parallel_for_loop(dummy, F const &f, I lim) {
                for (I i = 0; i < lim; ++i)
                    f(i);
            }

            template <class F, class I, class J>
            friend void thread_pool_parallel_for_loop(dummy, F const &f, I i_lim, J j_lim) {
                for (J j = 0; j < j_lim; ++j)
                    for (I i = 0; i < i_lim; ++i)
                        f(i, j);
            }

            template <class F, class I, class J, class K>
            friend void thread_pool_parallel_for_loop(dummy, F const &f, I i_lim, J j_lim, K k_lim) {
                for (K k = 0; k < k_lim; ++k)
                    for (J j = 0; j < j_lim; ++j)
                        for (I i = 0; i < i_lim; ++i)
                            f(i, j, k);
            }
        };
    } // namespace thread_pool
} // namespace gridtools
