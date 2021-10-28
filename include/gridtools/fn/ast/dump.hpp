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

        template <class I>
        struct sym {
            friend std::ostream &operator<<(std::ostream &s, sym) { return s << "f" << I::value; }
        };

        template <class Body, class Arity>
        struct fun {
            friend std::ostream &operator<<(std::ostream &s, fun) {
                return s << "[](" << params<Arity::value>() << ") {\n\treturn " << Body() << ";\n}";
            }
        };

        template <class T>
        struct full_parse {
            using type = T;
        };

        template <auto F, class... Args>
        using make_fun =
            fun<typename full_parse<parse<F, Args...>>::type, std::integral_constant<size_t, sizeof...(Args)>>;

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

        template <class Pass, class... Args>
        struct full_parse<builtin<builtins::scan, Pass, Args...>> {
            using type = builtin<builtins::scan,
                make_fun<Pass::value, void, void, void, Args...>,
                typename full_parse<Args>::type...>;
        };

        template <class Pass, class Init, class... Args>
        struct full_parse<builtin<builtins::reduce, Pass, Init, Args...>> {
            using type = builtin<builtins::reduce,
                make_fun<Pass::value, void, Args...>,
                make_fun<Init::value, Args...>,
                typename full_parse<Args>::type...>;
        };

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
            using fun_t = make_fun<F, Args...>;
            using tbl_t = meta::push_back<typename make_tbl<meta::first<fun_t>>::type, fun_t>;
            using funs_t = meta::transform<replace_sym_f<tbl_t>::template apply, tbl_t>;

            friend std::ostream &operator<<(std::ostream &s, dump_t) {
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
                s << "}\nusing " << Name << "_impl_::" << Name << ";\n";
                return s;
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
        auto &&name = boost::typeindex::type_id<Tag>().pretty_name();
        return s << name.substr(name.find_last_of(":") + 1) << "(" << seq<Args{}...> << ")";
    }

    template <class I, class Arg>
    std::ostream &operator<<(std::ostream &s, builtin<builtins::tuple_get, I, Arg>) {
        return s << "tuple_get<" << I::value << ">(" << Arg() << ")";
    }

    template <auto... Vs, class Arg>
    std::ostream &operator<<(std::ostream &s, builtin<builtins::shift, meta::val<Vs...>, Arg>) {
        using namespace dump_impl_;
        return s << "shift<" << seq<Vs...> << ">(" << Arg() << ")";
    }

    template <class Pass, class IsBackward, class... Args>
    std::ostream &operator<<(std::ostream &s, builtin<builtins::scan, Pass, IsBackward, Args...>) {
        using namespace dump_impl_;
        s << "scan<" << Pass();
        if (IsBackward::value)
            s << ", true";
        s << ">(" << seq<Args{}...> << ")";
        return s;
    }

    template <class F, class... Args>
    std::ostream &operator<<(std::ostream &s, builtin<builtins::tlift, F, Args...>) {
        using namespace dump_impl_;
        return s << "tlift<" << F() << ">(" << seq<Args{}...> << ")";
    }

    template <class F, class... Args>
    std::ostream &operator<<(std::ostream &s, builtin<builtins::ilift, F, Args...>) {
        using namespace dump_impl_;
        return s << "ilift<" << F() << ">(" << seq<Args{}...> << ")";
    }

    template <class F, class Init, class... Args>
    std::ostream &operator<<(std::ostream &s, builtin<builtins::reduce, F, Init, Args...>) {
        using namespace dump_impl_;
        return s << "reduce<" << F() << ", " << Init() << ">(" << seq<Args{}...> << ")";
    }
} // namespace gridtools::fn::ast
