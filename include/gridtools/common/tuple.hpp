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
/**
 * Implementation of a tuple with constexpr ctr and random element getter
 * Similar to std::tuple but GPU capable.
 * Notice: Current version is optimized for stornig integrals elements, returning elements by constexpr copy.
 * In order to use with more complex types (non constexprable), a version returning by ref should be used instead
 */
#pragma once
#include "./defs.hpp"
#include "./generic_metafunctions/gt_integer_sequence.hpp"
#include "./generic_metafunctions/meta.hpp"
#include "./host_device.hpp"
#include <utility>

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup tuple Tuple Implementation
        @{
    */

    template <class... Ts>
    struct tuple;

    template <class T>
    struct tuple_size : std::tuple_size<T> {};

    template <class... Ts>
    struct tuple_size<tuple<Ts...>> : std::integral_constant<size_t, sizeof...(Ts)> {};

    template <size_t I, class T>
    struct tuple_element : std::tuple_element<I, T> {};

    template <size_t I, class... Ts>
    struct tuple_element<I, tuple<Ts...>> : meta::at_c<tuple<Ts...>, I> {};

    namespace impl_ {
        template <size_t I, class T>
        struct tuple_entry {
            T m_value;

            GT_FUNCTION constexpr explicit tuple_entry(T const &value) : m_value(value) {}
            GT_FUNCTION constexpr tuple_entry() : m_value() {}

            GT_FUNCTION int swap(tuple_entry &other) {
                T tmp = m_value;
                m_value = static_cast<T &&>(other.m_value);
                other.m_value = static_cast<T &&>(tmp);
                return 0;
            }
        };

        template <class IndexSequence, class... Ts>
        struct tuple_impl;

        template <size_t... Is, class... Ts>
        struct tuple_impl<gt_index_sequence<Is...>, Ts...> : tuple_entry<Is, Ts>... {
            GT_FUNCTION constexpr explicit tuple_impl(Ts &&... ts) : tuple_entry<Is, Ts>(std::forward<Ts>(ts))... {}

            GT_FUNCTION constexpr tuple_impl() : tuple_entry<Is, Ts>()... {}

            GT_FUNCTION void swap(tuple_impl &other) { all(tuple_entry<Is, Ts>::swap(other)...); }

            template <class... Args>
            GT_FUNCTION static void all(Args...) {}
        };

    } // namespace impl_

    template <class... Ts>
    class tuple {
      private:
        using tuple_impl_t = impl_::tuple_impl<gt_index_sequence_for<Ts...>, Ts...>;
        tuple_impl_t m_impl;

        template <size_t I, class... TTs>
        friend constexpr typename tuple_element<I, tuple<TTs...>>::type const &get(tuple<TTs...> const &);
        template <size_t I, class... TTs>
        friend constexpr typename tuple_element<I, tuple<TTs...>>::type &get(tuple<TTs...> &);

      public:
        GT_FUNCTION constexpr explicit tuple(Ts &&... ts) : m_impl(std::forward<Ts>(ts)...) {}

        GT_FUNCTION constexpr tuple() : m_impl() {}

        GT_FUNCTION static constexpr size_t size() { return sizeof...(Ts); }

        GT_FUNCTION void swap(tuple &other) noexcept { m_impl.swap(other.m_impl); }
    };

    template <size_t I, class... Ts>
    GT_FUNCTION constexpr typename tuple_element<I, tuple<Ts...>>::type const &get(tuple<Ts...> const &t) {
        GRIDTOOLS_STATIC_ASSERT(I >= 0 && I < sizeof...(Ts), "out of bounds tuple access");
        using type = typename tuple_element<I, tuple<Ts...>>::type;
        return static_cast<impl_::tuple_entry<I, type> const &>(t.m_impl).m_value;
    }

    template <size_t I, class... Ts>
    GT_FUNCTION constexpr typename tuple_element<I, tuple<Ts...>>::type &get(tuple<Ts...> &t) {
        GRIDTOOLS_STATIC_ASSERT(I >= 0 && I < sizeof...(Ts), "out of bounds tuple access");
        using type = typename tuple_element<I, tuple<Ts...>>::type;
        return static_cast<impl_::tuple_entry<I, type> &>(t.m_impl).m_value;
    }

    template <class... Ts>
    GT_FUNCTION constexpr tuple<Ts...> make_tuple(Ts &&... ts) {
        return tuple<Ts...>(std::forward<Ts>(ts)...);
    }

    namespace impl_ {

        template <size_t I, class... Ts>
        GT_FUNCTION constexpr enable_if_t<(I == 0), bool> tuple_eq(tuple<Ts...> const &t1, tuple<Ts...> const &t2) {
            return true;
        }

        template <size_t I, class... Ts>
        GT_FUNCTION constexpr enable_if_t<(I > 0), bool> tuple_eq(tuple<Ts...> const &t1, tuple<Ts...> const &t2) {
            return tuple_eq<I - 1>(t1, t2) && get<I - 1>(t1) == get<I - 1>(t2);
        }

        template <size_t I, class... Ts>
        GT_FUNCTION constexpr enable_if_t<(I == 0), bool> tuple_lt(tuple<Ts...> const &t1, tuple<Ts...> const &t2) {
            return true;
        }

        template <size_t I, class... Ts>
        GT_FUNCTION constexpr enable_if_t<(I > 0), bool> tuple_lt(tuple<Ts...> const &t1, tuple<Ts...> const &t2) {
            return tuple_lt<I - 1>(t1, t2) || (!tuple_lt<I - 1>(t1, t2) & &get<I - 1>(t1) < get<I - 1>(t2));
        }

    } // namespace impl_

    template <class... Ts>
    GT_FUNCTION constexpr bool operator==(tuple<Ts...> const &t1, tuple<Ts...> const &t2) {
        return impl_::tuple_eq<sizeof...(Ts)>(t1, t2);
    }

    template <class... Ts>
    GT_FUNCTION constexpr bool operator<(tuple<Ts...> const &t1, tuple<Ts...> const &t2) {
        return impl_::tuple_lt<sizeof...(Ts)>(t1, t2);
    }

    template <class... Ts>
    GT_FUNCTION constexpr bool operator!=(tuple<Ts...> const &t1, tuple<Ts...> const &t2) {
        return !(t1 == t2);
    }

    template <class... Ts>
    GT_FUNCTION constexpr bool operator<=(tuple<Ts...> const &t1, tuple<Ts...> const &t2) {
        return (t1 < t2) || (t1 == t2);
    }

    template <class... Ts>
    GT_FUNCTION constexpr bool operator>(tuple<Ts...> const &t1, tuple<Ts...> const &t2) {
        return !(t1 <= t2);
    }

    template <class... Ts>
    GT_FUNCTION constexpr bool operator>=(tuple<Ts...> const &t1, tuple<Ts...> const &t2) {
        return !(t1 < t2);
    }

    template <class T>
    struct is_tuple : std::false_type {};

    template <class... Ts>
    struct is_tuple<tuple<Ts...>> : std::true_type {};

    /** @} */
    /** @} */

} // namespace gridtools
