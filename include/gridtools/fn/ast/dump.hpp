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

#include <sstream>
#include <string>

#include <boost/type_index.hpp>

#include "../../common/for_each.hpp"
#include "../../common/str_literal.hpp"

#include "../../meta.hpp"
#include "../builtins.hpp"
#include "../lambda.hpp"
#include "nodes.hpp"
#include "parse.hpp"

namespace gridtools {
    template <class T, T V>
    std::ostream &operator<<(std::ostream &s, integral_constant<T, V>) {
        return s << "integral_constant<" << boost::typeindex::type_id<T>().pretty_name() << ", " << V << ">()";
    }
    template <int V>
    std::ostream &operator<<(std::ostream &s, integral_constant<int, V>) {
        return s << V << "_c";
    }
} // namespace gridtools

namespace gridtools::fn::ast {
    namespace dump_impl_ {
        template <auto... Args>
        struct seq_t {
            friend std::ostream &operator<<(std::ostream &s, seq_t) {
                bool need_comma = false;
                (..., [&] {
                    if (need_comma)
                        s << ", ";
                    s << Args;
                    need_comma = true;
                }());
                return s;
            }
        };

        template <auto... Args>
        constexpr seq_t<Args...> seq = {};

        template <class I>
        struct sym {
            friend std::ostream &operator<<(std::ostream &s, sym) { return s << "f" << I::value; }
        };

        template <size_t N>
        struct params {
            friend std::ostream &operator<<(std::ostream &s, params) {
                for (size_t i = 0; i != N; ++i) {
                    if (i)
                        s << ", ";
                    s << "auto const &x" << i;
                }
                return s;
            }
        };

        template <class Body, class Arity>
        struct fun {
            friend std::ostream &operator<<(std::ostream &s, fun) {
                return s << "[](" << params<Arity::value>() << ") {\n\treturn " << Body() << ";\n}";
            }
        };

        template <class Tag>
        struct builtin_fun {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) {
                using namespace dump_impl_;
                auto &&name = boost::typeindex::type_id<Tag>().pretty_name();
                return s << name.substr(name.find_last_of(":") + 1);
            }
        };

        template <class F>
        struct builtin_fun<builtins::ilift<F>> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) { return s << "ilift<" << F() << ">"; }
        };

        template <class F>
        struct builtin_fun<builtins::tlift<F>> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) { return s << "tlift<" << F() << ">"; }
        };

        template <size_t I>
        struct builtin_fun<builtins::tuple_get<I>> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) { return s << "tuple_get<" << I << ">"; }
        };

        template <auto... Vs>
        struct builtin_fun<builtins::shift<Vs...>> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) { return s << "shift<" << seq<Vs...> << ">"; }
        };

        template <class IsBackward,
            class InitPass,
            class InitGet,
            class Pass,
            class Get,
            class Prologues,
            class Epilogues>
        struct builtin_fun<
            builtins::scan<IsBackward, meta::list<InitPass, InitGet>, meta::list<Pass, Get>, Prologues, Epilogues>> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) {
                s << "scan_" << (IsBackward::value ? "b" : "f") << "wd<" << InitPass() << ", " << Pass() << ", "
                  << InitGet() << ", " << Get() << ">";
                for_each<Prologues>(
                    [&]<class P, class G>(meta::list<P, G>) { s << ".prologue<" << P() << ", " << G() << ">"; });
                for_each<Epilogues>(
                    [&]<class P, class G>(meta::list<P, G>) { s << ".epilogue<" << P() << ", " << G() << ">"; });
                return s;
            }
        };

        template <class F, class Init>
        struct builtin_fun<builtins::reduce<F, Init>> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) {
                return s << "reduce<" << F() << ", " << Init() << ">";
            }
        };

        namespace lazy {
            template <class T>
            struct full_parse {
                using type = T;
            };

            template <class>
            struct make_seq;

            template <template <class...> class L, class... Args>
            struct make_seq<L<Args...>> {
                using type = seq_t<Args{}...>;
            };

            template <class Tree, class Arity>
            struct make_f {
                using type = fun<Tree, Arity>;
            };

            template <class Tag, class... Args, class Arity>
            struct make_f<builtin<Tag, Args...>, Arity> {
                using args_t = meta::list<Args...>;
                using params_t = meta::transform<in, meta::make_indices_for<args_t>>;
                using type =
                    meta::if_<std::is_same<params_t, args_t>, builtin_fun<Tag>, fun<builtin<Tag, Args...>, Arity>>;
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(full_parse, class T, T);
        GT_META_DELEGATE_TO_LAZY(make_seq, class T, T);

        template <auto F, class... Args>
        using make_fun =
            typename lazy::make_f<full_parse<parse<F, Args...>>, std::integral_constant<size_t, sizeof...(Args)>>::type;

        namespace lazy {
            template <template <class...> class Node, class... Trees>
            struct full_parse<Node<Trees...>> {
                using type = Node<typename full_parse<Trees>::type...>;
            };

            template <class F, class... Args>
            struct full_parse<lambda<F, Args...>> {
                using type = lambda<make_fun<F::value, Args...>, typename full_parse<Args>::type...>;
            };

            template <class F, class... Args>
            struct full_parse<tmp<F, Args...>> {
                using type = tmp<make_fun<F::value, Args...>, typename full_parse<Args>::type...>;
            };

            template <class F, class... Args>
            struct full_parse<inlined<F, Args...>> {
                using type = inlined<make_fun<F::value, Args...>, typename full_parse<Args>::type...>;
            };

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
            struct full_parse<builtin<builtins::scan<IsBackward,
                                          meta::val<InitPass, InitGet>,
                                          meta::val<Pass, Get>,
                                          meta::list<meta::val<ProloguePasses, PrologueGets>...>,
                                          meta::list<meta::val<EpiloguePasses, EpilogueGets>...>>,
                Args...>> {
                using type = builtin<builtins::scan<IsBackward,
                                         meta::list<make_fun<InitPass, Args...>, make_fun<InitGet, void>>,
                                         meta::list<make_fun<Pass, void, Args...>, make_fun<Get, void>>,
                                         meta::list<meta::list<make_fun<ProloguePasses, void, Args...>,
                                             make_fun<PrologueGets, void>>...>,
                                         meta::list<meta::list<make_fun<EpiloguePasses, void, Args...>,
                                             make_fun<EpilogueGets, void>>...>>,
                    typename full_parse<Args>::type...>;
            };

            template <class Pass, class Init, class... Args>
            struct full_parse<builtin<builtins::reduce<Pass, Init>, Args...>> {
                using type =
                    builtin<builtins::reduce<make_fun<Pass::value, void, Args...>, make_fun<Init::value, Args...>>,
                        typename full_parse<Args>::type...>;
            };
        } // namespace lazy

        template <class Tree>
        struct make_tbl {
            using type = meta::list<>;
        };

        template <template <class...> class Node, class... Trees>
        struct make_tbl<Node<Trees...>> {
            using type = meta::dedup<meta::concat<typename make_tbl<Trees>::type...>>;
        };

        template <class F, class Arity>
        struct make_tbl<fun<F, Arity>> {
            using type = meta::push_back<typename make_tbl<F>::type, fun<F, Arity>>;
        };

        template <class T, class Tbl>
        struct replace_fun {
            using type = T;
        };

        template <template <class...> class Node, class... Trees, class Tbl>
        struct replace_fun<Node<Trees...>, Tbl> {
            using type = Node<typename replace_fun<Trees, Tbl>::type...>;
        };

        template <class Tree, class Arity, class Tbl>
        struct replace_fun<fun<Tree, Arity>, Tbl> {
            using type = sym<typename meta::st_position<Tbl, fun<Tree, Arity>>::type>;
        };

        template <class T>
        struct get_body {
            using type = T;
        };

        template <class Body, class Arity>
        struct get_body<fun<Body, Arity>> {
            using type = Body;
        };

        template <class T, class Tbl>
        struct replace_fun_top {
            using type = typename replace_fun<T, Tbl>::type;
        };

        template <template <class...> class Node, class... Trees, class Tbl>
        struct replace_fun_top<Node<Trees...>, Tbl> {
            using type = Node<typename replace_fun<Trees, Tbl>::type...>;
        };

        template <str_literal Name, auto F, class... Args>
        struct dump_t {
            friend std::ostream &operator<<(std::ostream &s, dump_t) {
                using fun_t = make_fun<F, Args...>;
                using tbl_t = typename make_tbl<typename get_body<fun_t>::type>::type;
                if constexpr (meta::length<tbl_t>() == 0) {
                    return s << "inline constexpr auto " << Name << " = " << fun_t() << ";\n";
                } else {
                    constexpr auto replace = []<class Fun>(Fun)->typename replace_fun_top<Fun, tbl_t>::type {
                        return {};
                    };
                    s << "namespace " << Name << "_impl_ {\n";
                    size_t i = 0;
                    for_each<tbl_t>(
                        [&](auto f) { s << "inline constexpr auto f" << i++ << " = " << replace(f) << ";\n"; });
                    s << "inline constexpr auto " << Name << " = " << replace(fun_t()) << ";\n";
                    return s << "}\nusing " << Name << "_impl_::" << Name << ";\n";
                }
            }
        };
    } // namespace dump_impl_

    template <str_literal Name, auto F, class... Args>
    constexpr dump_impl_::dump_t<Name, F, Args...> dump = {};

    template <class I>
    std::ostream &operator<<(std::ostream &s, in<I>) {
        return s << "x" << I::value;
    }

    template <class F, class... Args>
    std::ostream &operator<<(std::ostream &s, lambda<F, Args...>) {
        using namespace dump_impl_;
        return s << "lambda<" << F() << ">(" << seq<Args{}...> << ")";
    }

    template <class Tag, class... Args>
    std::ostream &operator<<(std::ostream &s, builtin<Tag, Args...>) {
        using namespace dump_impl_;
        return s << builtin_fun<Tag>() << "(" << make_seq<meta::list<Args...>>() << ")";
    }
} // namespace gridtools::fn::ast
