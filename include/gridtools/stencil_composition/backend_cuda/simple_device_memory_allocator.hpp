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

#include <memory>
#include <vector>

#include "../../common/cuda_util.hpp"
#include "../sid/simple_ptr_holder.hpp"

namespace gridtools {
    namespace cuda {
        /**
         * @brief Allocator for CUDA device memory.
         */
        class simple_device_memory_allocator {
            std::vector<std::shared_ptr<void>> m_buffers;

            template <class LazyT>
            friend auto allocate(simple_device_memory_allocator &self, LazyT, size_t size) {
                auto buffer = cuda_util::cuda_malloc<typename LazyT::type>(size);
                auto res = sid::device::make_simple_ptr_holder(buffer.get());
                self.m_buffers.emplace_back(std::move(buffer));
                return res;
            }
        };
    } // namespace cuda
} // namespace gridtools
