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
#elif defined(GT_STORAGE_CPU_KFIRST)
#include <gridtools/storage/cpu_kfirst.hpp>
namespace {
    using storage_traits_t = gridtools::storage::cpu_kfirst;
}
#elif defined(GT_STORAGE_CPU_IFIRST)
#include <gridtools/storage/cpu_ifirst.hpp>
namespace {
    using storage_traits_t = gridtools::storage::cpu_ifirst;
}
#endif

namespace gridtools {
    namespace storage {
        struct cuda;
        struct cpu_kfirst;
        struct cpu_ifirst;

        cuda backend_storage_traits(cuda const &);
        cpu_kfirst backend_storage_traits(cpu_kfirst const &);
        cpu_ifirst backend_storage_traits(cpu_ifirst const &);

        timer_omp backend_timer_impl(cpu_kfirst const &);
        timer_omp backend_timer_impl(cpu_ifirst const &);
        timer_cuda backend_timer_impl(cuda const &);

        inline char const *backend_name(cpu_kfirst const &) { return "cpu_kfirst"; }
        inline char const *backend_name(cpu_ifirst const &) { return "cpu_ifirst"; }
        inline char const *backend_name(cuda const &) { return "cuda"; }
    } // namespace storage
} // namespace gridtools
