/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @brief provides a wrapper to execute a pure function on device returning a boolean
 * The function has to be passed as a functor with a static apply() method.
 */

#pragma once

#ifndef __CUDACC__
#error This is CUDA only header
#endif

#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/meta/type_traits.hpp>

namespace gridtools {
    namespace on_device {
        // Note that this specializations are required because if Fun is a nullary function, operator() of
        // integral_constant will be taken instead of explicit conversion
        template <class T, T (*Fun)()>
        __global__ void kernel(T *res, integral_constant<T (*)(), Fun>) {
            *res = Fun();
        }
        template <class T, T (*Fun)()>
        T exec_with_shared_memory(size_t shm_size, integral_constant<T (*)(), Fun> fun) {
            static_assert(std::is_trivially_copyable<T>::value, "");
            auto res = cuda_util::cuda_malloc<T>();
            kernel<<<1, 1, shm_size>>>(res.get(), fun);
            GT_CUDA_CHECK(cudaDeviceSynchronize());
            return cuda_util::from_clone(res);
        }

        template <class Res, class Fun, class... Args>
        __global__ void kernel(Res *res, Fun fun, Args... args) {
            *res = fun(args...);
        }
        template <class Fun, class... Args, class Res = decay_t<result_of_t<Fun(Args...)>>>
        Res exec_with_shared_memory(size_t shm_size, Fun fun, Args... args) {
            static_assert(!std::is_pointer<Fun>::value, "");
            static_assert(conjunction<negation<std::is_pointer<Args>>...>::value, "");
            static_assert(std::is_trivially_copyable<Res>::value, "");
            auto res = cuda_util::cuda_malloc<Res>();
            kernel<<<1, 1, shm_size>>>(res.get(), fun, args...);
            GT_CUDA_CHECK(cudaDeviceSynchronize());
            return cuda_util::from_clone(res);
        }

        template <class Fun, class... Args>
        auto exec(Fun &&fun, Args &&... args)
            GT_AUTO_RETURN(exec_with_shared_memory(0, std::forward<Fun>(fun), std::forward<Args>(args)...));
    } // namespace on_device
} // namespace gridtools
