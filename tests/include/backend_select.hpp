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

// stencil composition backend
#ifdef GT_BACKEND_X86
#define GT_STORAGE_X86
#define GT_TIMER_OMP
#include <gridtools/stencil_composition/backend/x86.hpp>
namespace {
    using backend_t = gridtools::x86::backend<>;
}
#elif defined(GT_BACKEND_NAIVE)
#define GT_STORAGE_X86
#define GT_TIMER_DUMMY
#include <gridtools/stencil_composition/backend/naive.hpp>
namespace {
    using backend_t = gridtools::naive::backend;
}
#elif defined(GT_BACKEND_MC)
#define GT_STORAGE_MC
#define GT_TIMER_OMP
#include <gridtools/stencil_composition/backend/mc.hpp>
namespace {
    using backend_t = gridtools::mc::backend;
}
#elif defined(GT_BACKEND_CUDA)
#define GT_STORAGE_CUDA
#define GT_TIMER_CUDA
#include <gridtools/stencil_composition/backend/cuda.hpp>
namespace {
    using backend_t = gridtools::cuda::backend<>;
}
#elif defined(GT_BACKEND_CUDA2)
#define GT_STORAGE_CUDA
#define GT_TIMER_CUDA
#include <gridtools/stencil_composition/backend/cuda2.hpp>
namespace {
    using backend_t = gridtools::cuda2::backend<>;
}
#endif

#include "storage_select.hpp"
#include "timer_select.hpp"

namespace gridtools {
    template <class T>
    std::true_type backend_supports_icosahedral(T);

    template <class T>
    std::true_type backend_supports_vertical_stencils(T);

    template <class T>
    std::is_same<decltype(backend_timer_impl(T())), timer_dummy> backend_skip_benchmark(T const &) {
        return {};
    }

    namespace x86 {
        template <class, class>
        struct backend;

        template <class I, class J>
        storage::x86 backend_storage_traits(backend<I, J>);

        template <class I, class J>
        timer_omp backend_timer_impl(backend<I, J>);

        template <class I, class J>
        char const *backend_name(backend<I, J> const &) {
            return "x86";
        }
    } // namespace x86

    namespace naive {
        struct backend;

        storage::x86 backend_storage_traits(backend);
        timer_dummy backend_timer_impl(backend);
        inline char const *backend_name(backend const &) { return "naive"; }
    } // namespace naive

    namespace mc {
        struct backend;

        storage::mc backend_storage_traits(backend);

        std::false_type backend_supports_icosahedral(backend);
        timer_omp backend_timer_impl(backend);
        inline char const *backend_name(backend const &) { return "mc"; }
    } // namespace mc

    namespace cuda {
        template <class, class, class>
        struct backend;

        template <class I, class J, class K>
        storage::cuda backend_storage_traits(backend<I, J, K>);

        template <class I, class J, class K>
        timer_cuda backend_timer_impl(backend<I, J, K>);

        template <class I, class J, class K>
        char const *backend_name(backend<I, J, K> const &) {
            return "cuda";
        }
    } // namespace cuda

    namespace cuda2 {
        template <class, class, class>
        struct backend;

        template <class I, class J, class K>
        storage::cuda backend_storage_traits(backend<I, J, K>);

        template <class I, class J, class K>
        std::false_type backend_supports_vertical_stencils(backend<I, J, K>);

        template <class I, class J, class K>
        timer_cuda backend_timer_impl(backend<I, J, K>);

        template <class I, class J, class K>
        char const *backend_name(backend<I, J, K> const &) {
            return "cuda2";
        }
    } // namespace cuda2
} // namespace gridtools
