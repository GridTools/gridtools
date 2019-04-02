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

#ifndef GT_ENABLE_METERS
#include "timer_dummy.hpp"
#else
#ifdef GT_USE_GPU
#include "timer_cuda.hpp"
#endif
#include "timer_omp.hpp"
#endif

namespace gridtools {
    template <typename T>
    struct timer_traits;

#ifndef GT_ENABLE_METERS
    template <typename T>
    struct timer_traits {
        using timer_type = timer_dummy;
    };
#else
#ifdef GT_USE_GPU
    template <>
    struct timer_traits<backend::cuda> {
        using timer_type = timer_cuda;
    };
#endif
    template <>
    struct timer_traits<backend::x86> {
        using timer_type = timer_omp;
    };
    template <>
    struct timer_traits<backend::naive> {
        using timer_type = timer_omp;
    };
    template <>
    struct timer_traits<backend::mc> {
        using timer_type = timer_omp;
    };
#endif
} // namespace gridtools
