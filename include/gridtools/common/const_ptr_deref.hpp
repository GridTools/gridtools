/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "host_device.hpp"

#ifdef GT_CUDACC
#include "cuda_type_traits.hpp"
#endif

namespace gridtools {

#if defined(GT_CUDACC) && defined(GT_CUDA_ARCH)

    template <class T>
    GT_FUNCTION constexpr std::enable_if_t<is_texture_type<T>::value, T> const_ptr_deref(T const *ptr) {
        return __ldg(ptr);
    }

#endif

    template <class T>
    GT_FUNCTION constexpr decltype(auto) const_ptr_deref(T &&ptr) {
        return *ptr;
    }

} // namespace gridtools
