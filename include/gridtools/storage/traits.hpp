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

#include "../meta.hpp"
#include "data_view.hpp"
#include "info.hpp"

namespace gridtools {
    namespace storage {
        namespace traits {
            namespace impl_ {}

            template <class Traits>
            constexpr bool is_host_referenceable =
                decltype(storage_is_host_referenceable(std::declval<Traits>()))::value;

            template <class Traits>
            constexpr size_t alignment = decltype(storage_alignment(std::declval<Traits>()))::value;

            template <class Traits, size_t Dims>
            using layout_type =
                decltype(storage_layout(std::declval<Traits>(), std::integral_constant<size_t, Dims>()));

            template <class Traits, class T>
            auto allocate(size_t size) {
                return storage_allocate(Traits(), meta::lazy::id<T>(), size);
            }

            template <class Traits, class T>
            using target_ptr_type = decltype(allocate<Traits, T>(0));

            template <class Traits, class T>
            std::enable_if_t<!is_host_referenceable<Traits>> update_target(T *dst, T const *src, size_t size) {
                storage_update_target(Traits(), dst, src, size);
            }

            template <class Traits, class T>
            std::enable_if_t<!is_host_referenceable<Traits>> update_host(T *dst, T const *src, size_t size) {
                storage_update_host(Traits(), dst, src, size);
            }

            template <class Traits, class T, size_t N, std::enable_if_t<is_host_referenceable<Traits>, int> = 0>
            auto storage_make_target_view(Traits, T *ptr, info<N> const &info) {
                return make_host_view(ptr, info);
            }

            template <class Traits, class T, size_t N>
            auto make_target_view(T *ptr, info<N> const &info) {
                return storage_make_target_view(Traits(), ptr, info);
            }
        } // namespace traits
    }     // namespace storage
} // namespace gridtools
