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

    namespace newapi {
        template <class T>
        concept Func = requires(T const x) {
            x.lambda;
        };

        template <class T>
        concept Value = requires(T const x) {
            x.template literal<0>;
            x.plus(x.template literal<0>, x.template literal<0>);
            x.minus(x.template literal<0>);
            x.minus(x.template literal<0>, x.template literal<0>);
            //            x.multiplies(x.template literal<0>, x.template literal<0>);
            //        x.divides;
            //        x.less;
            //        x.eq;
            //        x.not_;
            //        x.and_;
            //        x.or_;
        };

        struct rt_value {
            template <auto V>
            static constexpr auto literal = V;

            template <class Arg, class... Args>
            static constexpr decltype(auto) plus(Arg &&arg, Args &&... args) {
                return (std::forward<Arg>(arg) + ... + std::forward<Args>(args));
            };

            template <class Arg, class... Args>
            static constexpr decltype(auto) minus(Arg &&arg, Args &&... args) {
                if constexpr (sizeof...(Args))
                    return (std::forward<Arg>(arg) - ... - std::forward<Args>(args));
                else
                    return -std::forward<Arg>(arg);
            };
            template <class Arg, class... Args>
            static constexpr decltype(auto) multiplies(Arg &&arg, Args &&... args) {
                return (std::forward<Arg>(arg) * ... * std::forward<Args>(args));
            };
            template <class Tag, class... Args>
            static constexpr decltype(auto) builtin(Tag, Args &&... args){};
        };
        static_assert(Value<rt_value>);

        struct rt_func {
            static constexpr std::identity lambda = {};

            static constexpr auto buitin static constexpr lambda = {};
        };
        static_assert(Func<rt_func>);

        template <auto Val>
        constexpr auto literal = [](Value auto const &s) { return s.template literal<Val>; };
        constexpr auto plus = [](auto... args) {
            return [args...](Value auto const &s) { return s.plus(args(s)...); };
        };
        constexpr auto minus = [](auto... args) {
            return [args...](Value auto const &s) { return s.minus(args(s)...); };
        };
        //        constexpr auto multiplies = [](auto... args) {
        //            return [args...](Value auto const &s) { return s.multiplies(args(s)...); };
        //        };

        constexpr auto to_fun(auto val) {
            return [=](auto &&) { return val; };
        };

        template <auto F>
        constexpr auto lambda =
            [](Func auto const &s) { return s.lambda([&s](auto... args) { return F(to_fun(args)...)(s); }); };

        struct rt : rt_value, rt_func {};
        constexpr rt rt_v = {};

        constexpr auto expr = lambda<plus>;

        constexpr auto expr2 = lambda<[](auto x, auto y) { return plus(x, y); }>;

        static_assert(literal<1>(rt_value()) == 1);
        static_assert(plus(literal<1>, literal<2>)(rt_value()) == 3);
        static_assert(expr(rt_v)(1, 2) == 3);
        static_assert(expr2(rt_v)(1, 2) == 3);

        namespace newast {
            template <class...>
            struct literal {};
            template <class...>
            struct plus {};
            template <class...>
            struct minus {};
            template <class...>
            struct lambda {};

            template <class F>
            struct lam {
                F f;

                template <class... Args>
                lambda<decltype(f(Args)), Args...> operator()(Args...) {
                    return {};
                }
            };

        }; // namespace newast

        struct ast_parse {
            template <auto V>
            static constexpr newast::literal<meta::val<V>> literal = {};

            template <class... Ts>
            static constexpr newast::plus<Ts...> plus(Ts...) {
                return {};
            }
            template <class... Ts>
            static constexpr newast::minus<Ts...> minis(Ts...) {
                return {};
            }

            template <class F>
            static constexpr newast::lam<F> lambda(F f) {
                return {f};
            }
        };

        inline void foo() {
            int a = 1, b = 2, c = 3;
            assert(literal<1>(rt_value()) == 1);
            assert(plus(literal<1>, literal<2>)(rt_value()) == 3);
            assert(expr(rt())(a, b) == 3);
            assert(expr2(rt())(a, b) == 3);
        }
    }; // namespace newapi

    namespace builtins {
        struct deref {};
        struct can_deref {};
        template <auto...>
        struct shift {};
        template <class>
        struct ilift {};
        template <class>
        struct tlift {};
        template <class F, class Init>
        struct reduce {};
        struct plus {};
        struct minus {};
        struct multiplies {};
        struct divides {};
        struct make_tuple {};
        template <size_t>
        struct tuple_get {};
        struct if_ {};
        struct less {};
        struct eq {};
        struct not_ {};
        struct and_ {};
        struct or_ {};
        template <class IsForward, class Init, class Body, class Prologues, class Epilogues>
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
        constexpr decltype(auto) fn_default(builtins::shift<>, Arg &&arg) {
            return std::forward<Arg>(arg);
        }

        template <auto V, class Arg>
        constexpr auto fn_default(builtins::shift<V>, Arg const &arg) {
            return &tuple_util::get<V>(arg);
        }

        template <auto V0, auto V1, auto... Vs, class Arg>
        constexpr decltype(auto) fn_default(builtins::shift<V0, V1, Vs...>, Arg const &arg) {
            if constexpr (!std::is_same_v<undefined, decltype(fn_builtin(builtins::shift<V0, V1>(), arg))>)
                return builtin_fun(builtins::shift<Vs...>(), fn_builtin(builtins::shift<V0, V1>(), arg));
            else if constexpr (!std::is_same_v<undefined, decltype(fn_builtin(builtins::shift<V0>(), arg))>)
                return builtin_fun(builtins::shift<V1, Vs...>(), fn_builtin(builtins::shift<V0>(), arg));
        }

        template <auto F, class Args>
        struct lifted_iter {
            Args args;
        };

        template <auto... Offsets, auto F, class Args>
        constexpr auto fn_builtin(builtins::shift<Offsets...>, lifted_iter<F, Args> const &it) {
            auto args = tuple_util::transform(
                [](auto const &arg) { return builtin_fun(builtins::shift<Offsets...>(), arg); }, it.args);
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
        constexpr auto fn_default(builtins::ilift<F>, Args... args) {
            return lifted_iter<F::value, std::tuple<Args...>>{std::tuple(std::move(args)...)};
        }

        template <class T>
        constexpr meta::make_indices<tuple_util::size<T>, std::tuple> make_indices(T const &) {
            return {};
        }

        constexpr decltype(auto) get_offsets(auto const &arg, ...) { return offsets(arg); }

        template <class F, class Init, class... Args>
        constexpr auto fn_default(builtins::reduce<F, Init>, Args const &... args) {
            auto res =
                Init::value(decltype(builtin_fun(builtins::deref(), builtin_fun(builtins::shift<0>(), args))){}...);
            auto const &offsets = get_offsets(args...);
            tuple_util::for_each(
                [&]<class I>(int offset, I) {
                    if (offset != -1)
                        res = F::value(
                            res, builtin_fun(builtins::deref(), builtin_fun(builtins::shift<I::value>(), args))...);
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

        template <size_t I, class Arg>
        constexpr decltype(auto) fn_default(builtins::tuple_get<I>, Arg &&arg) {
            return tuple_util::get<I>(std::forward<Arg>(arg));
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
            auto InitPass,
            auto InitGet,
            auto Pass,
            auto Get,
            auto... ProloguePasses,
            auto... PrologueGets,
            auto... EpiloguePasses,
            auto... EpilogueGets,
            class... Args>
        constexpr decltype(auto) fn_default(builtins::scan<IsBackward,
                                                meta::val<InitPass, InitGet>,
                                                meta::val<Pass, Get>,
                                                meta::list<meta::val<ProloguePasses, PrologueGets>...>,
                                                meta::list<meta::val<EpiloguePasses, EpilogueGets>...>>,
            Args &&... args) {
            assert(false);
            return InitGet(InitPass(std::forward<Args>(args)...));
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

        template <bool IsBackward, class Init, class Body, class = meta::list<>, class = meta::list<>>
        struct scan_wrapper;

        template <bool IsBackward, class Init, class Body, class... Prologues, class... Epilogues>
        struct scan_wrapper<IsBackward, Init, Body, meta::list<Prologues...>, meta::list<Epilogues...>> {
            template <class... Args>
            decltype(auto) operator()(Args &&... args) const {
                return builtin_fun(builtins::scan<std::bool_constant<IsBackward>,
                                       Init,
                                       Body,
                                       meta::list<Prologues...>,
                                       meta::list<Epilogues...>>(),
                    std::forward<Args>(args)...);
            }

            template <auto Pass, auto Get = std::identity{}>
            static constexpr scan_wrapper<IsBackward,
                Init,
                Body,
                meta::list<Prologues..., meta::val<Pass, Get>>,
                meta::list<Epilogues...>>
                prologue = {};

            template <auto Pass, auto Get = std::identity{}>
            static constexpr scan_wrapper<IsBackward,
                Init,
                Body,
                meta::list<Prologues...>,
                meta::list<Epilogues..., meta::val<Pass, Get>>>
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
    constexpr auto tuple_get = builtin<builtins::tuple_get<I>>;

    template <auto... Vs>
    constexpr auto shift = builtin<builtins::shift<Vs...>>;

    template <auto F>
    constexpr auto ilift = builtin<builtins::ilift<meta::val<F>>>;

    template <auto F>
    constexpr auto tlift = builtin<builtins::tlift<meta::val<F>>>;

    template <auto F, bool UseTmp = false, class V = meta::val<F>>
    constexpr auto lift = builtin<std::conditional_t<UseTmp, builtins::tlift<V>, builtins::ilift<V>>>;

    template <auto F, auto Init>
    constexpr auto reduce = builtin<builtins::reduce<meta::val<F>, meta::val<Init>>>;

    template <auto InitPass, auto Pass, auto InitGet = std::identity{}, auto Get = std::identity{}>
    constexpr builtins_impl_::scan_wrapper<true, meta::val<InitPass, InitGet>, meta::val<Pass, Get>> scan_bwd{};
    template <auto InitPass, auto Pass, auto InitGet = std::identity{}, auto Get = std::identity{}>
    inline constexpr builtins_impl_::scan_wrapper<false, meta::val<InitPass, InitGet>, meta::val<Pass, Get>> scan_fwd{};
} // namespace gridtools::fn
