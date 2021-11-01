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

        template <class Tag, class... Args>
        struct builtin_fun {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) {
                using namespace dump_impl_;
                auto &&name = boost::typeindex::type_id<Tag>().pretty_name();
                return s << name.substr(name.find_last_of(":") + 1);
            }
            static constexpr size_t n_params = 0;
        };

        template <class F, class... Args>
        struct builtin_fun<builtins::ilift, F, Args...> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) { return s << "ilift<" << F() << ">"; }
            static constexpr size_t n_params = 1;
        };

        template <class F, class... Args>
        struct builtin_fun<builtins::tlift, F, Args...> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) { return s << "tlift<" << F() << ">"; }
            static constexpr size_t n_params = 1;
        };

        template <class I, class... Args>
        struct builtin_fun<builtins::tuple_get, I, Args...> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) {
                return s << "tuple_get<" << I::value << ">";
            }
            static constexpr size_t n_params = 1;
        };

        template <auto... Vs, class Arg>
        struct builtin_fun<builtins::shift, meta::val<Vs...>, Arg> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) { return s << "shift<" << seq<Vs...> << ">"; }
            static constexpr size_t n_params = 1;
        };

        template <class IsBackward, class Get, class Pass, class... Prologues, class... Epilogues, class... Args>
        struct builtin_fun<builtins::scan,
            IsBackward,
            Get,
            Pass,
            meta::list<Prologues...>,
            meta::list<Epilogues...>,
            Args...> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) {
                return s << "scan_" << (IsBackward::value ? "b" : "f") << "wd.pass<" << Pass() << ">.prologue<"
                         << seq<Prologues{}...> << ">.epilogue<" << seq<Epilogues{}...> << ">.get<" << Get() << ">";
            }
            static constexpr size_t n_params = 5;
        };

        template <class F, class Init, class... Args>
        struct builtin_fun<builtins::reduce, F, Init, Args...> {
            friend std::ostream &operator<<(std::ostream &s, builtin_fun) {
                return s << "reduce<" << F() << ", " << Init() << ">";
            }
            static constexpr size_t n_params = 2;
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
                using fun_t = builtin_fun<Tag, Args...>;
                using args_t = meta::drop_front_c<fun_t::n_params, meta::list<Args...>>;
                using params_t = meta::transform<in, meta::make_indices_for<args_t>>;
                using type = meta::if_<std::is_same<params_t, args_t>, fun_t, fun<builtin<Tag, Args...>, Arity>>;
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
                class Get,
                class Pass,
                auto Prologue,
                auto... Prologues,
                auto... Epilogues,
                class... Args>
            struct full_parse<builtin<builtins::scan,
                IsBackward,
                Get,
                Pass,
                meta::val<Prologue, Prologues...>,
                meta::val<Epilogues...>,
                Args...>> {
                using get_t = make_fun<Get::value, void>;
                using type = builtin<builtins::scan,
                    IsBackward,
                    make_fun<Get::value, void>,
                    make_fun<Pass::value, void, Args...>,
                    meta::list<make_fun<Prologue, Args...>, make_fun<Prologues, void, Args...>...>,
                    meta::list<make_fun<Epilogues, void, Args...>...>,
                    typename full_parse<Args>::type...>;
            };

            template <class Pass, class Init, class... Args>
            struct full_parse<builtin<builtins::reduce, Pass, Init, Args...>> {
                using type = builtin<builtins::reduce,
                    make_fun<Pass::value, void, Args...>,
                    make_fun<Init::value, Args...>,
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

        template <class Tbl>
        struct replace_sym_f {
            template <class Fun>
            using apply = fun<typename replace_fun<meta::first<Fun>, Tbl>::type, meta::second<Fun>>;
        };

        template <str_literal Name, auto F, class... Args>
        struct dump_t {
            friend std::ostream &operator<<(std::ostream &s, dump_t) {
                using fun_t = make_fun<F, Args...>;

                if constexpr (meta::is_instantiation_of<builtin_fun, fun_t>()) {
                    return s << "inline constexpr auto " << Name << " = " << fun_t() << ";\n";
                } else {
                    using tbl_t = meta::push_back<typename make_tbl<meta::first<fun_t>>::type, fun_t>;
                    using funs_t = meta::transform<replace_sym_f<tbl_t>::template apply, tbl_t>;
                    constexpr auto n = meta::length<funs_t>();
                    if constexpr (n == 1) {
                        return s << "inline constexpr auto " << Name << " = " << meta::first<funs_t>() << ";\n";
                    } else {
                        size_t i = 0;
                        s << "namespace " << Name << "_impl_ {\n";
                        for_each<funs_t>([&](auto fun) {
                            s << "inline constexpr auto ";
                            if (i == meta::length<funs_t>() - 1)
                                s << Name;
                            else
                                s << "f" << i++;
                            s << " = " << fun << ";\n";
                        });
                        return s << "}\nusing " << Name << "_impl_::" << Name << ";\n";
                    }
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
        using fun_t = builtin_fun<Tag, Args...>;
        using args_t = make_seq<meta::drop_front_c<fun_t::n_params, meta::list<Args...>>>;
        return s << fun_t() << "(" << args_t() << ")";
    }
} // namespace gridtools::fn::ast
