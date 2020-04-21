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
#ifndef GT_STORAGE_X86
#define GT_STORAGE_X86
#endif
#ifndef GT_TIMER_OMP
#define GT_TIMER_OMP
#endif
#include <gridtools/stencil_composition/backend/x86.hpp>
namespace {
    using backend_t = gridtools::x86::backend<>;
}
#elif defined(GT_BACKEND_NAIVE)
#ifndef GT_STORAGE_X86
#define GT_STORAGE_X86
#endif
#ifndef GT_TIMER_DUMMY
#define GT_TIMER_DUMMY
#endif
#include <gridtools/stencil_composition/backend/naive.hpp>
namespace {
    using backend_t = gridtools::naive::backend;
}
#elif defined(GT_BACKEND_MC)
#ifndef GT_STORAGE_MC
#define GT_STORAGE_MC
#endif
#ifndef GT_TIMER_OMP
#define GT_TIMER_OMP
#endif
#include <gridtools/stencil_composition/backend/mc.hpp>
namespace {
    using backend_t = gridtools::mc::backend<>;
}
#elif defined(GT_BACKEND_CUDA)
#ifndef GT_STORAGE_CUDA
#define GT_STORAGE_CUDA
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
#include <gridtools/stencil_composition/backend/cuda.hpp>
namespace {
    using backend_t = gridtools::cuda::backend<>;
}
#elif defined(GT_BACKEND_CUDA_HORIZONTAL)
#ifndef GT_STORAGE_CUDA
#define GT_STORAGE_CUDA
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
#include <gridtools/stencil_composition/backend/cuda_horizontal.hpp>
namespace {
    using backend_t = gridtools::cuda_horizontal::backend<>;
}
#endif

#include "storage_select.hpp"
#include "timer_select.hpp"

namespace gridtools {
    namespace x86 {
        template <class, class, class>
        struct backend;

        template <class I, class J, class T>
        storage::x86 backend_storage_traits(backend<I, J, T>);

        template <class I, class J, class T>
        timer_omp backend_timer_impl(backend<I, J, T>);

        template <class I, class J, class T>
        char const *backend_name(backend<I, J, T> const &) {
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
        template <class>
        struct backend;

        template <class T>
        storage::mc backend_storage_traits(backend<T>);

        template <class T>
        std::false_type backend_supports_icosahedral(backend<T>);

        template <class T>
        timer_omp backend_timer_impl(backend<T>);

        template <class T>
        char const *backend_name(backend<T> const &) {
            return "mc";
        }
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

    namespace cuda_horizontal {
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
            return "cuda_horizontal";
        }
    } // namespace cuda_horizontal
} // namespace gridtools
