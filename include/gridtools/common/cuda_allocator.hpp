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

#include "cuda_util.hpp"
#include <memory>

namespace gridtools {

    /**
     * @brief Allocator for CUDA device memory.
     */
    class cuda_allocator {
      public:
        /**
         * \param bytes Size of requested allocation in bytes.
         */
        std::shared_ptr<void> allocate(size_t bytes) {
            char *ptr;
            GT_CUDA_CHECK(cudaMalloc(&ptr, bytes));
            return std::shared_ptr<void>(ptr, [](char *ptr) { GT_CUDA_CHECK(cudaFree(ptr)); });
        }
    };
} // namespace gridtools
