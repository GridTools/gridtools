/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Shared with icosahedral test

#include <memory>
#include <ostream>
#include <vector>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil_composition/sid/simple_ptr_holder.hpp>

namespace gridtools {
    namespace {
        class simple_host_memory_allocator {
            std::vector<std::shared_ptr<void>> m_ptrs;

          public:
            template <class T>
            sid::host::simple_ptr_holder<T *> allocate(size_t num_elements) {
                T *ptr = new T[num_elements];
                m_ptrs.emplace_back(ptr, [](T *ptr) { delete[] ptr; });
                return {static_cast<T *>(m_ptrs.back().get())};
            }
        };

        template <class Tag, class T = typename Tag::type>
        sid::host::simple_ptr_holder<T *> allocate(simple_host_memory_allocator &alloc, Tag, size_t num_elements) {
            return alloc.template allocate<T>(num_elements);
        }
    } // namespace
} // namespace gridtools
