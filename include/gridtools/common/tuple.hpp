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

#include <type_traits>

#include "../meta/type_traits.hpp"
#include "../meta/utility.hpp"
#include "defs.hpp"
#include "generic_metafunctions/utility.hpp"
#include "host_device.hpp"

namespace gridtools {

    namespace impl_ {
        struct gt_tuple_getter;

        template <size_t I, class T, bool = std::is_empty<T>::value>
        class tuple_leaf {
            T m_value;
            friend gt_tuple_getter;

          protected:
            tuple_leaf(tuple_leaf const &) = default;
            tuple_leaf(tuple_leaf &&) = default;
            tuple_leaf &operator=(tuple_leaf const &) = default;
            tuple_leaf &operator=(tuple_leaf &&) = default;

            constexpr GT_FUNCTION tuple_leaf() noexcept : m_value() {}

            template <class Arg, enable_if_t<std::is_constructible<T, Arg &&>::value, int> = 0>
            constexpr GT_FUNCTION tuple_leaf(Arg &&arg) noexcept : m_value(const_expr::forward<Arg>(arg)) {}
        };

        template <size_t I, class T>
        class tuple_leaf<I, T, true> : T {
            friend gt_tuple_getter;

          protected:
            tuple_leaf() = default;
            tuple_leaf(tuple_leaf const &) = default;
            tuple_leaf(tuple_leaf &&) = default;
            tuple_leaf &operator=(tuple_leaf const &) = default;
            tuple_leaf &operator=(tuple_leaf &&) = default;

            template <class Arg, enable_if_t<std::is_constructible<T, Arg &&>::value, int> = 0>
            constexpr GT_FUNCTION tuple_leaf(Arg &&arg) noexcept : T(const_expr::forward<Arg>(arg)) {}
        };

        struct gt_tuple_getter {
            template <size_t I, class T>
            static constexpr GT_FUNCTION T const &get(tuple_leaf<I, T, false> const &obj) noexcept {
                return obj.m_value;
            }

            template <size_t I, class T>
            static GT_FUNCTION T &get(tuple_leaf<I, T, false> &obj) noexcept {
                return obj.m_value;
            }

            template <size_t I, class T>
            static constexpr GT_FUNCTION T &&get(tuple_leaf<I, T, false> &&obj) noexcept {
                return static_cast<T &&>(obj.m_value);
            }

            template <size_t I, class T>
            static constexpr GT_FUNCTION T const &get(tuple_leaf<I, T, true> const &obj) noexcept {
                return obj;
            }

            template <size_t I, class T>
            static GT_FUNCTION T &get(tuple_leaf<I, T, true> &obj) noexcept {
                return obj;
            }

            template <size_t I, class T>
            static constexpr GT_FUNCTION T &&get(tuple_leaf<I, T, true> &&obj) noexcept {
                return static_cast<T &&>(obj);
            }
        };

        template <class Indices, class... Ts>
        struct tuple_impl;

        template <size_t... Is, class... Ts>
        struct tuple_impl<meta::index_sequence<Is...>, Ts...> : tuple_leaf<Is, Ts>... {
            tuple_impl() = default;
            tuple_impl(tuple_impl const &) = default;
            tuple_impl(tuple_impl &&) = default;
            tuple_impl &operator=(tuple_impl const &) = default;
            tuple_impl &operator=(tuple_impl &&) = default;

          protected:
            template <class... Args>
            constexpr GT_FUNCTION tuple_impl(Args &&... args) noexcept
                : tuple_leaf<Is, Ts>(const_expr::forward<Args>(args))... {}

            template <class Src>
            constexpr GT_FUNCTION tuple_impl(Src &&src) noexcept
                : tuple_leaf<Is, Ts>(gt_tuple_getter::get<Is>(const_expr::forward<Src>(src)))... {}

            GT_FORCE_INLINE void swap(tuple_impl &other) noexcept {
                using std::swap;
                void((int[]){(swap(gt_tuple_getter::get<Is>(*this), gt_tuple_getter::get<Is>(other)), 0)...});
            }

            template <class... Args,
                enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                conjunction<std::is_assignable<Ts &, Args const &>...>::value,
                    int> = 0>
            GT_FUNCTION void assign(tuple_impl<meta::index_sequence<Is...>, Args...> const &src) noexcept {
                void((int[]){(gt_tuple_getter::get<Is>(*this) = gt_tuple_getter::get<Is>(src), 0)...});
            }

            template <class... Args,
                enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                conjunction<std::is_assignable<Ts &, Args &&>...>::value,
                    int> = 0>
            GT_FUNCTION void assign(tuple_impl<meta::index_sequence<Is...>, Args...> &&src) noexcept {
                void((int[]){(gt_tuple_getter::get<Is>(*this) = gt_tuple_getter::get<Is>(std::move(src)), 0)...});
            }
        };
    } // namespace impl_

    /**
     *  Simplified host/device aware implementation of std::tuple interface.
     *
     *  Nuances
     *  =======
     *
     *  - get/tuple_element/tuple_size, comparision operators etc. are not implemented. Instead `tuple` is adopted
     *    to use with tuple_util library.
     *  - `allocator` aware constructors are not implemented
     *  - all constructors are implicit. [which violates the Standard]
     *  - element wise direct constructor is not sfinae friendly
     *  - encapsulation is compromised over simplicity. In particular `tuple` has public inheritance from its
     * `tuple_impl`.
     *  - all methods declared as noexcept [which violates the Standard]
     *  - `swap` is implemented as a `__host__` function because it can call `std::swap`
     *
     */

    template <class... Ts>
    struct tuple : impl_::tuple_impl<meta::index_sequence_for<Ts...>, Ts...> {
        tuple() = default;
        tuple(tuple const &) = default;
        tuple(tuple &&) = default;
        tuple &operator=(tuple const &) = default;
        tuple &operator=(tuple &&) = default;

        constexpr GT_FUNCTION tuple(Ts const &... args) noexcept : tuple::tuple_impl(args...) {}

        template <class... Args,
            enable_if_t<sizeof...(Ts) == sizeof...(Args) && conjunction<std::is_constructible<Ts, Args &&>...>::value,
                int> = 0>
        constexpr GT_FUNCTION tuple(Args &&... args) noexcept : tuple::tuple_impl(const_expr::forward<Args>(args)...) {}

        template <class... Args,
            enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                            conjunction<std::is_constructible<Ts, Args const &>...>::value,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple<Args...> const &src) noexcept : tuple::tuple_impl(src) {}

        template <class... Args,
            enable_if_t<sizeof...(Ts) == sizeof...(Args) && conjunction<std::is_constructible<Ts, Args &&>...>::value,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple<Args...> &&src) noexcept : tuple::tuple_impl(const_expr::move(src)) {}

        using tuple::tuple_impl::swap;

        template <class Other>
        GT_FUNCTION auto operator=(Other &&other) GT_AUTO_RETURN((assign(std::forward<Other>(other)), *this));
    };

    template <class T>
    struct tuple<T> : impl_::tuple_leaf<0, T> {
        tuple() = default;
        tuple(tuple const &) = default;
        tuple(tuple &&) = default;
        tuple &operator=(tuple const &) = default;
        tuple &operator=(tuple &&) = default;

        constexpr GT_FUNCTION tuple(T const &arg) noexcept : tuple::tuple_leaf(arg) {}

        template <class Arg, enable_if_t<std::is_constructible<T, Arg &&>::value, int> = 0>
        constexpr GT_FUNCTION tuple(Arg &&arg) noexcept : tuple::tuple_leaf(const_expr::forward<Arg>(arg)) {}

        template <class Arg,
            enable_if_t<std::is_constructible<T, Arg const &>::value &&
                            !std::is_convertible<tuple<Arg> const &, T>::value &&
                            !std::is_constructible<T, tuple<Arg> const &>::value && !std::is_same<T, Arg>::value,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple<Arg> const &src) noexcept
            : tuple::tuple_leaf(impl_::gt_tuple_getter::get<0>(src)) {}

        template <class Arg,
            enable_if_t<std::is_constructible<T, Arg &&>::value && !std::is_convertible<tuple<Arg>, T>::value &&
                            !std::is_constructible<T, tuple<Arg>>::value && !std::is_same<T, Arg>::value,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple<Arg> &&src) noexcept
            : tuple::tuple_leaf(impl_::gt_tuple_getter::get<0>(const_expr::move(src))) {}

        GT_FORCE_INLINE void swap(tuple &other) noexcept {
            using std::swap;
            swap(impl_::gt_tuple_getter::get<0>(*this), impl_::gt_tuple_getter::get<0>(other));
        }

        template <class Arg, enable_if_t<std::is_assignable<T &, Arg const &>::value, int> = 0>
        GT_FUNCTION tuple &operator=(tuple<Arg> const &src) noexcept {
            impl_::gt_tuple_getter::get<0>(*this) = impl_::gt_tuple_getter::get<0>(src);
            return *this;
        }

        template <class Arg, enable_if_t<std::is_assignable<T &, Arg &&>::value, int> = 0>
        GT_FUNCTION tuple &operator=(tuple<Arg> &&src) noexcept {
            impl_::gt_tuple_getter::get<0>(*this) = impl_::gt_tuple_getter::get<0>(std::move(src));
            return *this;
        }
    };

    template <>
    struct tuple<> {
        GT_FORCE_INLINE void swap(tuple &) noexcept {}
    };

    template <class... Ts>
    GT_FORCE_INLINE void swap(tuple<Ts...> &lhs, tuple<Ts...> &rhs) noexcept {
        lhs.swap(rhs);
    }

    template <class... Ts>
    impl_::gt_tuple_getter tuple_getter(tuple<Ts...>);

} // namespace gridtools
