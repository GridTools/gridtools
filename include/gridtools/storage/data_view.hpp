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

#include <utility>

#include "../common/array.hpp"
#include "info.hpp"

namespace gridtools {
    namespace storage {
        template <class T, size_t N>
        struct host_view {
            T *m_ptr;
            storage::info<N> const *m_info;

            constexpr auto const &info() const { return *m_info; }
            constexpr auto length() const { return m_info->length(); }
            constexpr auto const &lengths() const { return m_info->lengths(); }
            auto *data() const { return m_ptr; }

            template <class... Args>
            auto operator()(Args &&... args) const -> decltype(m_ptr[m_info->index(std::forward<Args>(args)...)]) {
                return m_ptr[m_info->index(std::forward<Args>(args)...)];
            }

            decltype(auto) operator()(array<int, N> const &arg) const { return m_ptr[m_info->index(arg)]; }
        };

        template <class T, size_t N>
        host_view<T, N> make_host_view(T *ptr, info<N> const &info) {
            return {ptr, &info};
        }
    } // namespace storage
} // namespace gridtools
