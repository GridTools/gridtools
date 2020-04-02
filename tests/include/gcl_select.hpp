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

#include <gridtools/communication/low_level/gcl_arch.hpp>

#if defined(GT_GCL_GPU)
#ifndef GT_STORAGE_CUDA
#define GT_STORAGE_CUDA
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
namespace {
    using gcl_arch_t = gridtools::gcl_gpu;
}
#elif defined(GT_GCL_CPU)
#ifndef GT_STORAGE_X86
#define GT_STORAGE_X86
#endif
#ifndef GT_TIMER_OMP
#define GT_TIMER_OMP
#endif
namespace {
    using gcl_arch_t = gridtools::gcl_cpu;
}
#endif

#include "storage_select.hpp"
#include "timer_select.hpp"

namespace gridtools {
    storage::x86 backend_storage_traits(gcl_cpu const &);
    timer_omp backend_timer_impl(gcl_cpu const &);
    inline char const *backend_name(gcl_cpu const &) { return "cpu"; }

    storage::cuda backend_storage_traits(gcl_gpu const &);
    timer_cuda backend_timer_impl(gcl_gpu const &);
    inline char const *backend_name(gcl_gpu const &) { return "gpu"; }
} // namespace gridtools
