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

#include <cuda_runtime.h>
#include <memory>
#include <vector>

#include "../../common/cuda_util.hpp"
#include "../sid/simple_ptr_holder.hpp"

namespace gridtools {

    /**
     * @brief Allocator for CUDA device memory.
     */
    class simple_device_memory_allocator {
        std::vector<std::shared_ptr<void>> m_ptrs;

      public:
        template <class T>
        sid::device::simple_ptr_holder<T *> allocate(size_t num_elements) {
            T *ptr;
            GT_CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * num_elements));
            m_ptrs.emplace_back(ptr, [](T *ptr) { assert(cudaFree(ptr) == cudaSuccess); });
            return {ptr};
        }
    };

    template <class Tag, class T = typename Tag::type>
    sid::device::simple_ptr_holder<T *> allocate(simple_device_memory_allocator &alloc, Tag, size_t num_elements) {
        return alloc.template allocate<T>(num_elements);
    }
} // namespace gridtools
