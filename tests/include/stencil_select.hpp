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

// stencil backend
#if defined(GT_STENCIL_CPU_KFIRST)
#ifndef GT_STORAGE_CPU_KFIRST
#define GT_STORAGE_CPU_KFIRST
#endif
#ifndef GT_TIMER_OMP
#define GT_TIMER_OMP
#endif
#include <gridtools/stencil/cpu_kfirst.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::cpu_kfirst<>;
}
#elif defined(GT_STENCIL_NAIVE)
#ifndef GT_STORAGE_CPU_KFIRST
#define GT_STORAGE_CPU_KFIRST
#endif
#ifndef GT_TIMER_DUMMY
#define GT_TIMER_DUMMY
#endif
#include <gridtools/stencil/naive.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::naive;
}
#elif defined(GT_STENCIL_CPU_IFIRST)
#ifndef GT_STORAGE_CPU_IFIRST
#define GT_STORAGE_CPU_IFIRST
#endif
#ifndef GT_TIMER_OMP
#define GT_TIMER_OMP
#endif
#include <gridtools/stencil/cpu_ifirst.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::cpu_ifirst<>;
}
#elif defined(GT_STENCIL_CUDA)
#ifndef GT_STORAGE_CUDA
#define GT_STORAGE_CUDA
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
#include <gridtools/stencil/cuda.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::cuda<>;
}
#elif defined(GT_STENCIL_CUDA_HORIZONTAL)
#ifndef GT_STORAGE_CUDA
#define GT_STORAGE_CUDA
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
#include <gridtools/stencil/cuda_horizontal.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::cuda_horizontal<>;
}
#endif

#include "storage_select.hpp"
#include "timer_select.hpp"

namespace gridtools {
    namespace stencil {

        struct naive;
        storage::cpu_kfirst backend_storage_traits(naive);
        timer_dummy backend_timer_impl(naive);
        inline char const *backend_name(naive const &) { return "naive"; }

        namespace cpu_kfirst_backend {
            template <class, class, class>
            struct cpu_kfirst;

            template <class I, class J, class T>
            storage::cpu_kfirst backend_storage_traits(cpu_kfirst<I, J, T>);

            template <class I, class J, class T>
            timer_omp backend_timer_impl(cpu_kfirst<I, J, T>);

            template <class I, class J, class T>
            char const *backend_name(cpu_kfirst<I, J, T> const &) {
                return "cpu_kfirst";
            }
        } // namespace cpu_kfirst_backend

        namespace cpu_ifirst_backend {
            template <class>
            struct cpu_ifirst;

            template <class T>
            storage::cpu_ifirst backend_storage_traits(cpu_ifirst<T>);

            template <class T>
            std::false_type backend_supports_icosahedral(cpu_ifirst<T>);

            template <class T>
            timer_omp backend_timer_impl(cpu_ifirst<T>);

            template <class T>
            char const *backend_name(cpu_ifirst<T> const &) {
                return "cpu_ifirst";
            }
        } // namespace cpu_ifirst_backend

        namespace cuda_backend {
            template <class, class, class>
            struct cuda;

            template <class I, class J, class K>
            storage::cuda backend_storage_traits(cuda<I, J, K>);

            template <class I, class J, class K>
            timer_cuda backend_timer_impl(cuda<I, J, K>);

            template <class I, class J, class K>
            char const *backend_name(cuda<I, J, K> const &) {
                return "cuda";
            }
        } // namespace cuda_backend

        namespace cuda_horizontal_backend {
            template <class, class, class>
            struct cuda_horizontal;

            template <class I, class J, class K>
            storage::cuda backend_storage_traits(cuda_horizontal<I, J, K>);

            template <class I, class J, class K>
            std::false_type backend_supports_vertical_stencils(cuda_horizontal<I, J, K>);

            template <class I, class J, class K>
            timer_cuda backend_timer_impl(cuda_horizontal<I, J, K>);

            template <class I, class J, class K>
            char const *backend_name(cuda_horizontal<I, J, K> const &) {
                return "cuda_horizontal";
            }
        } // namespace cuda_horizontal_backend
    }     // namespace stencil
} // namespace gridtools
