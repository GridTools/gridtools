/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/**
 * @file
 * Some c++14/c++17 utility drop offs. Please refer to C++14/17 specifications
 * to know more about them.
 */

#include <cstddef>
#include <utility>

namespace gridtools {
    namespace meta {
        template <typename Int, Int... Indices>
        struct integer_sequence {
            using value_type = Int;
            static constexpr std::size_t size() noexcept { return sizeof...(Indices); }
        };

        namespace _impl {
            template <typename Seq, std::size_t Size, size_t Rem>
            struct expand_integer_sequence;

            template <typename Int, Int... Is, std::size_t Size>
            struct expand_integer_sequence<integer_sequence<Int, Is...>, Size, 0> {
                using type = integer_sequence<Int, Is..., (Size + Is)...>;
            };

            template <typename Int, Int... Is, std::size_t Size>
            struct expand_integer_sequence<integer_sequence<Int, Is...>, Size, 1> {
                using type = integer_sequence<Int, Is..., (Size + Is)..., 2 * Size>;
            };

            template <typename Int, std::size_t N>
            struct generate_integer_sequence {
                using type = typename expand_integer_sequence<typename generate_integer_sequence<Int, N / 2>::type,
                    N / 2,
                    N % 2>::type;
            };

            template <typename Int>
            struct generate_integer_sequence<Int, 0> {
                using type = integer_sequence<Int>;
            };
        } // namespace _impl

        template <typename Int, Int N>
        using make_integer_sequence = typename _impl::generate_integer_sequence<Int, N>::type;

        template <std::size_t... Indices>
        using index_sequence = integer_sequence<std::size_t, Indices...>;

        template <std::size_t N>
        using make_index_sequence = make_integer_sequence<std::size_t, N>;

        template <class... Ts>
        using index_sequence_for = make_index_sequence<sizeof...(Ts)>;
    } // namespace meta
} // namespace gridtools
