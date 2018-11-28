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

    namespace tuple_detail {
        struct getter;

        template <size_t I, class T, bool = std::is_empty<T>::value>
        class leaf {
            T m_value;
            friend getter;

          protected:
            leaf(leaf const &) = default;
            leaf(leaf &&) = default;
            leaf &operator=(leaf const &) = default;
            leaf &operator=(leaf &&) = default;

            constexpr GT_FUNCTION leaf() noexcept : m_value() {}

            template <class Arg, enable_if_t<std::is_constructible<T, Arg &&>::value, int> = 0>
            constexpr GT_FUNCTION leaf(Arg &&arg) noexcept : m_value(const_expr::forward<Arg>(arg)) {}
        };

        template <size_t I, class T>
        class leaf<I, T, true> : T {
            friend getter;

          protected:
            leaf() = default;
            leaf(leaf const &) = default;
            leaf(leaf &&) = default;
            leaf &operator=(leaf const &) = default;
            leaf &operator=(leaf &&) = default;

            template <class Arg, enable_if_t<std::is_constructible<T, Arg &&>::value, int> = 0>
            constexpr GT_FUNCTION leaf(Arg &&arg) noexcept : T(const_expr::forward<Arg>(arg)) {}
        };

        struct getter {
            template <size_t I, class T>
            static constexpr GT_FUNCTION T const &get(leaf<I, T, false> const &obj) noexcept {
                return obj.m_value;
            }

            template <size_t I, class T>
            static GT_FUNCTION T &get(leaf<I, T, false> &obj) noexcept {
                return obj.m_value;
            }

            template <size_t I, class T>
            static constexpr GT_FUNCTION T &&get(leaf<I, T, false> &&obj) noexcept {
                return static_cast<T &&>(obj.m_value);
            }

            template <size_t I, class T>
            static constexpr GT_FUNCTION T const &get(leaf<I, T, true> const &obj) noexcept {
                return obj;
            }

            template <size_t I, class T>
            static GT_FUNCTION T &get(leaf<I, T, true> &obj) noexcept {
                return obj;
            }

            template <size_t I, class T>
            static constexpr GT_FUNCTION T &&get(leaf<I, T, true> &&obj) noexcept {
                return static_cast<T &&>(obj);
            }
        };

        template <class Indices, class... Ts>
        struct impl;

        template <size_t... Is, class... Ts>
        struct impl<meta::index_sequence<Is...>, Ts...> : leaf<Is, Ts>... {
            impl() = default;
            impl(impl const &) = default;
            impl(impl &&) = default;
            impl &operator=(impl const &) = default;
            impl &operator=(impl &&) = default;

            constexpr GT_FUNCTION impl(Ts const &... args) noexcept : leaf<Is, Ts>(args)... {}
            constexpr GT_FUNCTION impl(Ts &&... args) noexcept : leaf<Is, Ts>(const_expr::move(args))... {}

            template <class... Args,
                enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                conjunction<std::is_constructible<Ts, Args &&>...>::value,
                    int> = 0>
            constexpr GT_FUNCTION impl(Args &&... args) noexcept : leaf<Is, Ts>(const_expr::forward<Args>(args))... {}

            template <class... Args,
                enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                conjunction<std::is_constructible<Ts, Args const &>...>::value,
                    int> = 0>
            constexpr GT_FUNCTION impl(impl<meta::index_sequence<Is...>, Args...> const &src) noexcept
                : leaf<Is, Ts>(getter::get<Is>(src))... {}

            template <class... Args,
                enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                conjunction<std::is_constructible<Ts, Args &&>...>::value,
                    int> = 0>
            constexpr GT_FUNCTION impl(impl<meta::index_sequence<Is...>, Args...> &&src) noexcept
                : leaf<Is, Ts>(getter::get<Is>(const_expr::move(src)))... {}

            GT_FORCE_INLINE void swap(impl &other) noexcept {
                using std::swap;
                void((int[]){(swap(getter::get<Is>(*this), getter::get<Is>(other)), 0)...});
            }

          protected:
            template <class... Args,
                enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                conjunction<std::is_assignable<Ts &, Args const &>...>::value,
                    int> = 0>
            GT_FUNCTION void assign(impl<meta::index_sequence<Is...>, Args...> const &src) noexcept {
                void((int[]){(getter::get<Is>(*this) = getter::get<Is>(src), 0)...});
            }

            template <class... Args,
                enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                conjunction<std::is_assignable<Ts &, Args &&>...>::value,
                    int> = 0>
            GT_FUNCTION void assign(impl<meta::index_sequence<Is...>, Args...> &&src) noexcept {
                void((int[]){(getter::get<Is>(*this) = getter::get<Is>(std::move(src)), 0)...});
            }
        };
    } // namespace tuple_detail

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
     *  - encapsulation is compromised over simplicity. In particular `tuple` has public inheritance from its `impl`.
     *  - all methods declared as noexcept [which violates the Standard]
     *  - `swap` is implemented as a `__host__` function because it can call `std::swap`
     *
     */

    template <class... Ts>
    struct tuple : tuple_detail::impl<meta::index_sequence_for<Ts...>, Ts...> {
        tuple() = default;
        tuple(tuple const &) = default;
        tuple(tuple &&) = default;
        tuple &operator=(tuple const &) = default;
        tuple &operator=(tuple &&) = default;

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ <= 10 || defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1800
        template <class... Args,
            enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                            conjunction<std::is_constructible<Ts, Args const &>...>::value,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple_detail::impl<meta::index_sequence_for<Ts...>, Args...> const &src) noexcept
            : tuple::impl(src) {}

        template <class... Args,
            enable_if_t<sizeof...(Ts) == sizeof...(Args) && conjunction<std::is_constructible<Ts, Args &&>...>::value,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple_detail::impl<meta::index_sequence_for<Ts...>, Args...> &&src) noexcept
            : tuple::impl(const_expr::move(src)) {}

        using tuple_detail::impl<meta::index_sequence_for<Ts...>, Ts...>::impl;
#else
        using tuple::impl::impl;
#endif
        using tuple::impl::swap;

        template <class Other>
        GT_FUNCTION auto operator=(Other &&other) GT_AUTO_RETURN((assign(std::forward<Other>(other)), *this));
    };

    template <class T>
    struct tuple<T> : tuple_detail::leaf<0, T> {
        tuple() = default;
        tuple(tuple const &) = default;
        tuple(tuple &&) = default;
        tuple &operator=(tuple const &) = default;
        tuple &operator=(tuple &&) = default;

        constexpr GT_FUNCTION tuple(T const &arg) noexcept : tuple::leaf(arg) {}

        template <class Arg, enable_if_t<std::is_constructible<T, Arg &&>::value, int> = 0>
        constexpr GT_FUNCTION tuple(Arg &&arg) noexcept : tuple::leaf(const_expr::forward<Arg>(arg)) {}

        template <class Arg,
            enable_if_t<std::is_constructible<T, Arg const &>::value &&
                            !std::is_convertible<tuple<Arg> const &, T>::value &&
                            !std::is_constructible<T, tuple<Arg> const &>::value && !std::is_same<T, Arg>::value,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple<Arg> const &src) noexcept : tuple::leaf(tuple_detail::getter::get<0>(src)) {}

        template <class Arg,
            enable_if_t<std::is_constructible<T, Arg &&>::value && !std::is_convertible<tuple<Arg>, T>::value &&
                            !std::is_constructible<T, tuple<Arg>>::value && !std::is_same<T, Arg>::value,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple<Arg> &&src) noexcept
            : tuple::leaf(tuple_detail::getter::get<0>(const_expr::move(src))) {}

        GT_FORCE_INLINE void swap(tuple &other) noexcept {
            using std::swap;
            swap(tuple_detail::getter::get<0>(*this), tuple_detail::getter::get<0>(other));
        }

        template <class Arg, enable_if_t<std::is_assignable<T &, Arg const &>::value, int> = 0>
        GT_FUNCTION tuple &operator=(tuple<Arg> const &src) noexcept {
            tuple_detail::getter::get<0>(*this) = tuple_detail::getter::get<0>(src);
            return *this;
        }

        template <class Arg, enable_if_t<std::is_assignable<T &, Arg &&>::value, int> = 0>
        GT_FUNCTION tuple &operator=(tuple<Arg> &&src) noexcept {
            tuple_detail::getter::get<0>(*this) = tuple_detail::getter::get<0>(std::move(src));
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
    tuple_detail::getter tuple_getter(tuple<Ts...>);

} // namespace gridtools
