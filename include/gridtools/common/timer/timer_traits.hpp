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

#include "../defs.hpp"

#include "timer.hpp"

#include "timer_cuda.hpp"
#include "timer_omp.hpp"

namespace gridtools {
    template <typename T>
    struct timer_traits;

    template <class... Params>
    struct timer_traits<cuda::backend<Params...>> {
        using timer_type = timer<timer_cuda>;
    };
    template <class... Params>
    struct timer_traits<x86::backend<Params...>> {
        using timer_type = timer<timer_omp>;
    };
    template <>
    struct timer_traits<backend::naive> {
        using timer_type = timer<timer_omp>;
    };
    template <>
    struct timer_traits<backend::mc> {
        using timer_type = timer<timer_omp>;
    };
} // namespace gridtools
