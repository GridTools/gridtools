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

#include <type_traits>

#include <gridtools/meta.hpp>

// fn backend
#if defined(GT_FN_NAIVE)
#ifndef GT_STENCIL_NAIVE
#define GT_STENCIL_NAIVE
#endif
#ifndef GT_STORAGE_CPU_KFIRST
#define GT_STORAGE_CPU_KFIRST
#endif
#ifndef GT_TIMER_DUMMY
#define GT_TIMER_DUMMY
#endif
#include <gridtools/fn/backend2/naive.hpp>
namespace {
    using fn_backend_t = gridtools::fn::backend::naive;
}
#elif defined(GT_FN_GPU)
#ifndef GT_STENCIL_GPU
#define GT_STENCIL_GPU
#endif
#ifndef GT_STORAGE_GPU
#define GT_STORAGE_GPU
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
#include <gridtools/fn/backend2/gpu.hpp>
namespace {
    using fn_backend_t = gridtools::fn::backend::gpu;
}
#endif

#include "stencil_select.hpp"
#include "storage_select.hpp"
#include "timer_select.hpp"

namespace gridtools::fn::backend {
    namespace naive_impl_ {
        struct naive;
        storage::cpu_kfirst backend_storage_traits(naive);
        timer_dummy backend_timer_impl(naive);
        inline char const *backend_name(naive const &) { return "naive"; }
    } // namespace naive_impl_

    namespace gpu_impl_ {
        template <class>
        struct gpu;
        template <class BlockSizes>
        storage::gpu backend_storage_traits(gpu<BlockSizes>);
        template <class BlockSizes>
        timer_cuda backend_timer_impl(gpu<BlockSizes>);
        template <class BlockSizes>
        inline char const *backend_name(gpu<BlockSizes> const &) {
            return "gpu";
        }
    } // namespace gpu_impl_
} // namespace gridtools::fn::backend
