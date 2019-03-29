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
#include "functional.hpp"
#include <memory>
#include <vector>

namespace gridtools {

    /**
     * @brief Allocator for CUDA device memory.
     */
    class cuda_allocator {
        std::vector<std::shared_ptr<void>> m_ptrs;

      public:
        /**
         * \param bytes Size of requested allocation in bytes.
         */
        template <class T>
        host_device::constant<T *> allocate(size_t num_elements) {
            T *ptr;
            GT_CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * num_elements));
            m_ptrs.emplace_back(ptr, [](T *ptr) { GT_CUDA_CHECK(cudaFree(ptr)); });
            return {static_cast<T *>(m_ptrs.back().get())};
        }
    };
} // namespace gridtools
