/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cassert>

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../common/integral_constant.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "offsets.hpp"

namespace gridtools::fn {
    namespace builtins {
        struct deref {};
        struct can_deref {};
        struct shift {};
        struct ilift {};
        struct tlift {};
        struct reduce {};
        struct plus {};
        struct minus {};
        struct multiplies {};
        struct divides {};
        struct make_tuple {};
        struct tuple_get {};
        struct if_ {};
        struct less {};
        struct eq {};
        struct not_ {};
        struct and_ {};
        struct or_ {};
        struct scan {};
    }; // namespace builtins

    namespace builtins_impl_ {

        struct undefined {};

        undefined fn_builtin(...);

        template <class Tag, class... Args>
        constexpr decltype(auto) builtin_fun(Tag tag, Args &&... args);

        template <class Arg>
        constexpr decltype(auto) fn_default(builtins::deref, Arg &&arg) {
            return *std::forward<Arg>(arg);
        }

        template <class Arg>
        constexpr decltype(auto) fn_default(builtins::can_deref, Arg const &arg) {
            if constexpr (std::is_constructible_v<bool, Arg>)
                return bool(arg);
            else
                return true;
        }

        template <class Arg>
        constexpr decltype(auto) fn_default(builtins::shift, meta::val<>, Arg &&arg) {
            return std::forward<Arg>(arg);
        }

        template <auto V, class Arg>
        constexpr auto fn_default(builtins::shift, meta::val<V>, Arg const &arg) {
            return &tuple_util::get<V>(arg);
        }

        template <auto V0, auto V1, auto... Vs, class Arg>
        constexpr decltype(auto) fn_default(builtins::shift, meta::val<V0, V1, Vs...>, Arg const &arg) {
            if constexpr (!std::is_same_v<undefined,
                              decltype(fn_builtin(builtins::shift(), meta::constant<V0, V1>, arg))>)
                return builtin_fun(builtins::shift(),
                    meta::constant<Vs...>,
                    fn_builtin(builtins::shift(), meta::constant<V0, V1>, arg));
            else if constexpr (!std::is_same_v<undefined,
                                   decltype(fn_builtin(builtins::shift(), meta::constant<V0>, arg))>)
                return builtin_fun(builtins::shift(),
                    meta::constant<V1, Vs...>,
                    fn_builtin(builtins::shift(), meta::constant<V0>, arg));
        }

        template <auto F, class Args>
        struct lifted_iter {
            Args args;
        };

        template <class Offsets, auto F, class Args>
        constexpr auto fn_builtin(builtins::shift, Offsets, lifted_iter<F, Args> const &it) {
            auto args = tuple_util::transform(
                [](auto const &arg) { return builtin_fun(builtins::shift(), Offsets(), arg); }, it.args);
            return lifted_iter<F, decltype(args)>{std::move(args)};
        }

        template <auto F, class Args>
        constexpr auto fn_builtin(builtins::deref, lifted_iter<F, Args> const &it) {
            return std::apply(F, it.args);
        }

        template <auto F, class Arg, class... Args>
        constexpr decltype(auto) fn_offsets(lifted_iter<F, std::tuple<Arg, Args...>> const &it) {
            return fn_offsets(std::get<0>(it.args));
        }

        template <class F, class... Args>
        constexpr auto fn_default(builtins::ilift, F, Args... args) {
            return lifted_iter<F::value, std::tuple<Args...>>{std::tuple(std::move(args)...)};
        }

        template <class T>
        constexpr meta::make_indices<tuple_util::size<T>, std::tuple> make_indices(T const &) {
            return {};
        }

        constexpr decltype(auto) get_offsets(auto const &arg, ...) { return offsets(arg); }

        template <class F, class Init, class... Args>
        constexpr auto fn_default(builtins::reduce, F, Init, Args const &... args) {
            auto res = Init::value(
                decltype(builtin_fun(builtins::deref(), builtin_fun(builtins::shift(), meta::constant<0>, args))){}...);
            auto const &offsets = get_offsets(args...);
            tuple_util::for_each(
                [&]<class I>(int offset, I) {
                    if (offset != -1)
                        res = F::value(res,
                            builtin_fun(
                                builtins::deref(), builtin_fun(builtins::shift(), meta::constant<I::value>, args))...);
                },
                offsets,
                make_indices(offsets));
            return res;
        }

        template <class... Args>
        constexpr decltype(auto) fn_default(builtins::plus, Args &&... args) {
            return (... + std::forward<Args>(args));
        }

        template <class Arg>
        constexpr decltype(auto) fn_default(builtins::minus, Arg &&arg) {
            return -std::forward<Arg>(arg);
        }

        template <class... Args>
        constexpr decltype(auto) fn_default(builtins::minus, Args &&... args) {
            return (... - std::forward<Args>(args));
        }

        template <class... Args>
        constexpr decltype(auto) fn_default(builtins::multiplies, Args &&... args) {
            return (... * std::forward<Args>(args));
        }

        template <class... Args>
        constexpr decltype(auto) fn_default(builtins::divides, Args &&... args) {
            return (... / std::forward<Args>(args));
        }

        template <class... Args>
        constexpr decltype(auto) fn_default(builtins::make_tuple, Args &&... args) {
            return std::tuple(std::forward<Args>(args)...);
        }

        template <class I, class Arg>
        constexpr decltype(auto) fn_default(builtins::tuple_get, I, Arg &&arg) {
            return tuple_util::get<I::value>(std::forward<Arg>(arg));
        }

        template <class C, class L, class R>
        constexpr decltype(auto) fn_default(builtins::if_, C const &c, L &&l, R &&r) {
            if constexpr (!is_integral_constant<C>())
                return c ? std::forward<L>(l) : std::forward<R>(r);
            else if constexpr (C::value)
                return std::forward<L>(l);
            else
                return std::forward<R>(r);
        }

        template <class L, class R>
        constexpr decltype(auto) fn_default(builtins::less, L &&l, R &&r) {
            return std::forward<L>(l) < std::forward<R>(r);
        }

        template <class L, class R>
        constexpr decltype(auto) fn_default(builtins::eq, L &&l, R &&r) {
            return std::forward<L>(l) == std::forward<R>(r);
        }

        template <class T>
        constexpr decltype(auto) fn_default(builtins::not_, T &&val) {
            return !std::forward<T>(val);
        }

        template <class... Args>
        constexpr decltype(auto) fn_default(builtins::and_, Args &&... args) {
            return (... && std::forward<Args>(args));
        }

        template <class... Args>
        constexpr decltype(auto) fn_default(builtins::or_, Args &&... args) {
            return (... || std::forward<Args>(args));
        }

        template <class IsBackward,
            auto Get,
            auto Pass,
            auto Prologue,
            auto... Prologues,
            auto... Epilogues,
            class... Args>
        constexpr decltype(auto) fn_default(builtins::scan,
            IsBackward,
            meta::val<Get>,
            meta::val<Pass>,
            meta::val<Prologue, Prologues...>,
            meta::val<Epilogues...>,
            Args &&... args) {
            assert(false);
            return Get(Prologue(std::forward<Args>(args)...));
        }

        template <class Tag, class... Args>
        constexpr decltype(auto) builtin_fun(Tag tag, Args &&... args) {
            if constexpr (std::is_same_v<decltype(fn_builtin(tag, std::forward<Args>(args)...)), undefined>)
                return builtins_impl_::fn_default(tag, std::forward<Args>(args)...);
            else
                return fn_builtin(tag, std::forward<Args>(args)...);
        }

        template <class Tag>
        constexpr auto builtin = []<class... Args>(Args &&... args) -> decltype(auto) {
            return builtin_fun(Tag(), std::forward<Args>(args)...);
        };

        template <class IsBackward,
            class Get = meta::val<std::identity{}>,
            class Pass = void,
            class Prologue = void,
            class Epilogue = meta::val<>>
        struct scan_wrapper {
            template <class... Args>
            decltype(auto) operator()(Args &&... args) const {
                return builtin_fun(
                    builtins::scan(), IsBackward(), Get(), Pass(), Prologue(), Epilogue(), std::forward<Args>(args)...);
            }

            template <auto F>
            static constexpr scan_wrapper<IsBackward,
                Get,
                std::enable_if_t<std::is_void_v<Pass>, meta::val<F>>,
                Prologue,
                Epilogue>
                pass = {};

            template <auto F>
            static constexpr scan_wrapper<IsBackward,
                std::enable_if_t<std::is_same_v<Get, meta::val<std::identity{}>>, meta::val<F>>,
                Pass,
                Prologue,
                Epilogue>
                get = {};

            template <auto F, auto... Fs>
            static constexpr scan_wrapper<IsBackward,
                Get,
                Pass,
                std::enable_if_t<std::is_void_v<Prologue>, meta::val<F, Fs...>>,
                Epilogue>
                prologue = {};

            template <auto F, auto... Fs>
            static constexpr scan_wrapper<IsBackward,
                Get,
                Pass,
                Prologue,
                std::enable_if_t<std::is_same_v<Epilogue, meta::val<>>, meta::val<F, Fs...>>>
                epilogue = {};
        };

    } // namespace builtins_impl_
    using builtins_impl_::builtin;

    inline constexpr auto deref = builtin<builtins::deref>;
    inline constexpr auto can_deref = builtin<builtins::can_deref>;
    inline constexpr auto plus = builtin<builtins::plus>;
    inline constexpr auto minus = builtin<builtins::minus>;
    inline constexpr auto divides = builtin<builtins::divides>;
    inline constexpr auto multiplies = builtin<builtins::multiplies>;
    inline constexpr auto make_tuple = builtin<builtins::make_tuple>;
    inline constexpr auto if_ = builtin<builtins::if_>;
    inline constexpr auto less = builtin<builtins::less>;
    inline constexpr auto eq = builtin<builtins::eq>;
    inline constexpr auto not_ = builtin<builtins::not_>;
    inline constexpr auto and_ = builtin<builtins::and_>;
    inline constexpr auto or_ = builtin<builtins::or_>;

    template <size_t I>
    constexpr auto tuple_get = std::bind_front(builtin<builtins::tuple_get>, std::integral_constant<size_t, I>());

    template <auto... Vs>
    constexpr auto shift = std::bind_front(builtin<builtins::shift>, meta::constant<Vs...>);

    template <auto F>
    constexpr auto ilift = std::bind_front(builtin<builtins::ilift>, meta::constant<F>);

    template <auto F>
    constexpr auto tlift = std::bind_front(builtin<builtins::tlift>, meta::constant<F>);

    template <auto F, bool UseTmp = false>
    constexpr auto lift = std::bind_front(
        builtin<std::conditional_t<UseTmp, builtins::tlift, builtins::ilift>>, meta::constant<F>);

    template <auto F, auto Init>
    constexpr auto reduce = std::bind_front(builtin<builtins::reduce>, meta::constant<F>, meta::constant<Init>);

    inline constexpr builtins_impl_::scan_wrapper<std::true_type> scan_bwd{};
    inline constexpr builtins_impl_::scan_wrapper<std::false_type> scan_fwd{};
} // namespace gridtools::fn
