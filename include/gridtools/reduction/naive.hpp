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

#include <cstdlib>

namespace gridtools {
    namespace reduction {
        struct naive {};

        template <class F, class T>
        T reduction_reduce(naive, F f, T const *buff, size_t n) {
            T res = buff[0];
            for (size_t i = 1; i != n; i++)
                res = f(res, buff[i]);
            return res;
        }

        inline size_t reduction_round_size(naive, size_t size) { return size; }
        inline size_t reduction_allocation_size(naive, size_t size) { return size; }

        template <class T>
        void reduction_fill(naive,
            T const &initial_value,
            T *ptr,
            size_t /*data_size*/,
            size_t /*rounded_size*/,
            size_t allocation_size,
            bool has_holes) {
            for (size_t i = 0; i != allocation_size; ++i)
                ptr[i] = initial_value;
        }
    } // namespace reduction
} // namespace gridtools
