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

#include "../../common/cuda_util.hpp"
#include "../../common/integral_constant.hpp"
#include "../sid/allocator.hpp"

namespace gridtools {
    namespace cuda {
        /**
         * @brief Allocator for CUDA device memory.
         */
        using simple_device_memory_allocator =
            sid::device::allocator<GT_INTEGRAL_CONSTANT_FROM_VALUE(&cuda_util::cuda_malloc<char>)>;

    } // namespace cuda
} // namespace gridtools
