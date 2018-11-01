/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#pragma once

/**
 * @file
 * Some c++14/c++17 utility drop offs. Please refer to C++14/17 specifications
 * to know more about them.
 */

#include <utility>

namespace gridtools {
    namespace meta {
        template <typename Int, Int... Indices>
        struct integer_sequence {
            using value_type = Int;
            static constexpr size_t size() noexcept { return sizeof...(Indices); }
        };

        namespace _impl {
            template <typename Seq, size_t Size, size_t Rem>
            struct expand_integer_sequence;

            template <typename Int, Int... Is, size_t Size>
            struct expand_integer_sequence<integer_sequence<Int, Is...>, Size, 0> {
                using type = integer_sequence<Int, Is..., (Size + Is)...>;
            };

            template <typename Int, Int... Is, size_t Size>
            struct expand_integer_sequence<integer_sequence<Int, Is...>, Size, 1> {
                using type = integer_sequence<Int, Is..., (Size + Is)..., 2 * Size>;
            };

            template <typename Int, size_t N>
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

        template <size_t... Indices>
        using index_sequence = integer_sequence<size_t, Indices...>;

        template <size_t N>
        using make_index_sequence = make_integer_sequence<size_t, N>;

        template <class... Ts>
        using index_sequence_for = make_index_sequence<sizeof...(Ts)>;
    } // namespace meta
} // namespace gridtools
