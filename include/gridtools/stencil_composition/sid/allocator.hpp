/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef GT_TARGET_ITERATING
// DON'T USE #pragma once HERE!!!
#ifndef GT_STENCIL_COMPOSITION_SID_ALLOCATOR_HPP_
#define GT_STENCIL_COMPOSITION_SID_ALLOCATOR_HPP_

#include <map>
#include <memory>
#include <stack>
#include <utility>
#include <vector>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../meta.hpp"
#include "simple_ptr_holder.hpp"

namespace gridtools {
    namespace sid {
        namespace allocator_impl_ {

            template <class Impl, class Ptr = decltype(std::declval<Impl const>()(size_t{}))>
            struct cached_proxy_f;

            template <class Impl, class T, class Deleter>
            struct cached_proxy_f<Impl, std::unique_ptr<T, Deleter>> {
                using ptr_t = std::unique_ptr<T, Deleter>;
                using stack_t = std::stack<ptr_t>;

                struct deleter_f {
                    using pointer = typename ptr_t::pointer;
                    Deleter m_deleter;
                    stack_t &m_stack;

                    void operator()(pointer ptr) const { m_stack.emplace(ptr, m_deleter); }
                };
                using cached_ptr_t = std::unique_ptr<T, deleter_f>;

                Impl m_impl;

                cached_ptr_t operator()(size_t size) const {
                    static std::map<size_t, stack_t> stack_map;
                    auto &stack = stack_map[size];
                    ptr_t ptr;
                    if (stack.empty()) {
                        ptr = m_impl(size);
                    } else {
                        ptr = std::move(stack.top());
                        stack.pop();
                    }
                    return {ptr.release(), {ptr.get_deleter(), stack}};
                }
            };
        } // namespace allocator_impl_
    }     // namespace sid
} // namespace gridtools

#define GT_FILENAME <gridtools/stencil_composition/sid/allocator.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif
#else

namespace gridtools {
    namespace sid {
        GT_TARGET_NAMESPACE {
            template <class Impl, class Ptr = decltype(std::declval<Impl const>()(size_t{}))>
            class allocator;

            template <class Impl, class T, class Deleter>
            class allocator<Impl, std::unique_ptr<T, Deleter>> {
                Impl m_impl;
                std::vector<std::unique_ptr<T, Deleter>> m_buffers;

              public:
                allocator() = default;
                allocator(Impl impl) : m_impl(std::move(impl)) {}

                template <class LazyT>
                friend auto allocate(allocator &self, LazyT, size_t size) {
                    using type = typename LazyT::type;
                    auto ptr = self.m_impl(sizeof(type) * size);
                    self.m_buffers.push_back(self.m_impl(sizeof(type) * size));
                    return make_simple_ptr_holder(reinterpret_cast<type *>(self.m_buffers.back().get()));
                }
            };

            template <class Impl>
            struct cached_allocator : allocator<allocator_impl_::cached_proxy_f<Impl>> {
                cached_allocator() = default;
                cached_allocator(Impl impl) : cached_allocator::allocator({std::move(impl)}) {}
            };

            template <class Impl>
            allocator<Impl> make_allocator(Impl impl) {
                return {std::move(impl)};
            }

            template <class Impl>
            cached_allocator<Impl> make_cached_allocator(Impl impl) {
                return {std::move(impl)};
            }
        }
    } // namespace sid
} // namespace gridtools

#endif
