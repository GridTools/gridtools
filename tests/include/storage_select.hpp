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

#include "timer_select.hpp"

#if defined(GT_STORAGE_CUDA)
#include <gridtools/storage/cuda.hpp>
namespace {
    using storage_traits_t = gridtools::storage::cuda;
}
#elif defined(GT_STORAGE_X86)
#include <gridtools/storage/x86.hpp>
namespace {
    using storage_traits_t = gridtools::storage::x86;
}
#elif defined(GT_STORAGE_MC)
#include <gridtools/storage/mc.hpp>
namespace {
    using storage_traits_t = gridtools::storage::mc;
}
#endif

namespace gridtools {
    namespace storage {
        struct cuda;
        struct x86;
        struct mc;

        cuda backend_storage_traits(cuda const &);
        x86 backend_storage_traits(x86 const &);
        mc backend_storage_traits(mc const &);

        timer_omp backend_timer_impl(x86 const &);
        timer_omp backend_timer_impl(mc const &);
        timer_cuda backend_timer_impl(cuda const &);

        inline char const *backend_name(x86 const &) { return "x86"; }
        inline char const *backend_name(mc const &) { return "mc"; }
        inline char const *backend_name(cuda const &) { return "cuda"; }
    } // namespace storage
} // namespace gridtools
