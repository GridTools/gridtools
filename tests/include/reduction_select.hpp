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

#include <type_traits>

#include <gridtools/meta.hpp>

// reduction backend
#if defined(GT_REDUCTION_NAIVE)
#ifndef GT_STENCIL_NAIVE
#define GT_STENCIL_NAIVE
#endif
#ifndef GT_STORAGE_CPU_KFIRST
#define GT_STORAGE_CPU_KFIRST
#endif
#ifndef GT_TIMER_DUMMY
#define GT_TIMER_DUMMY
#endif
#include <gridtools/reduction/naive.hpp>
namespace {
    using reduction_backend_t = gridtools::reduction::naive;
}
#elif defined(GT_REDUCTION_GPU)
#ifndef GT_STENCIL_GPU
#define GT_STENCIL_GPU
#endif
#ifndef GT_STORAGE_GPU
#define GT_STORAGE_GPU
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
#include <gridtools/reduction/gpu.hpp>
namespace {
    using reduction_backend_t = gridtools::reduction::gpu;
}
#endif

#include "stencil_select.hpp"
#include "storage_select.hpp"
#include "timer_select.hpp"

namespace gridtools {
    namespace reduction {
        struct naive;
        storage::cpu_kfirst backend_storage_traits(naive);
        timer_dummy backend_timer_impl(naive);
        inline char const *backend_name(naive const &) { return "naive"; }

        namespace gpu_backend {
            struct gpu;
            storage::gpu backend_storage_traits(gpu);
            timer_cuda backend_timer_impl(gpu);
            inline char const *backend_name(gpu const &) { return "gpu"; }
        } // namespace gpu_backend
    }     // namespace reduction
} // namespace gridtools
