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

#include "../../meta.hpp"
#include "generate.hpp"
#include "nodes.hpp"
#include "parse.hpp"

namespace gridtools::fn::ast {
    namespace tmp_impl_ {
        namespace lazy {
            template <class Leaf>
            struct normailze_shifts {
                using type = Leaf;
            };

            template <template <class...> class Node, class... Trees>
            struct normailze_shifts<Node<Trees...>> {
                using type = Node<typename normailze_shifts<Trees>::type...>;
            };

            template <class Tree, auto... Outer, auto... Inner>
            struct normailze_shifts<shifted<shifted<Tree, meta::val<Inner...>>, meta::val<Outer...>>> {
                using type = typename normailze_shifts<shifted<Tree, meta::val<Inner..., Outer...>>>::type;
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(normailze_shifts, class T, T);

        template <class>
        struct has_tmps : std::false_type {};

        template <template <class...> class Node, class... Trees>
        struct has_tmps<Node<Trees...>> : std::disjunction<has_tmps<Trees>...> {};

        template <auto F, class... Trees>
        struct has_tmps<lambda<meta::val<F>, Trees...>> : has_tmps<decltype(F(Trees()...))> {};

        template <class F, class... Trees>
        struct has_tmps<inlined<F, Trees...>> : has_tmps<lambda<F, Trees...>> {};

        template <class F, class... Trees>
        struct has_tmps<tmp<F, Trees...>> : std::true_type {};

        template <class T>
        using is_arg =
            std::bool_constant<meta::is_instantiation_of<tmp, T>::value || meta::is_instantiation_of<in, T>::value>;

        template <class>
        struct collect_args {
            using type = meta::list<>;
        };

        template <template <class...> class Node, class... Trees>
        struct collect_args<Node<Trees...>> {
            using type = meta::dedup<meta::concat<typename collect_args<Trees>::type...>>;
        };

        template <class V>
        struct collect_args<in<V>> {
            using type = meta::list<in<V>>;
        };

        template <class... Ts>
        struct collect_args<tmp<Ts...>> {
            using type = meta::list<tmp<Ts...>>;
        };

        template <class...>
        struct remap_args;

        template <class Map, class Leaf>
        struct remap_args<Map, Leaf> {
            using type = Leaf;
        };

        template <class Map, template <class...> class Node, class... Trees>
        struct remap_args<Map, Node<Trees...>> {
            using type = Node<typename remap_args<Map, Trees>::type...>;
        };

        template <class Map, class V>
        struct remap_args<Map, in<V>> {
            using type = in<meta::second<meta::mp_find<Map, in<V>>>>;
        };

        template <class Map, class... Ts>
        struct remap_args<Map, tmp<Ts...>> {
            using type = in<meta::second<meta::mp_find<Map, tmp<Ts...>>>>;
        };

        template <class...>
        struct collapse;

        template <template <class...> class Node, class Tree>
        struct collapse_impl {
            using old_args_t = typename collect_args<Tree>::type;
            using map_t = meta::zip<old_args_t, meta::make_indices_for<old_args_t>>;
            using fun_t = meta::val<generate<typename remap_args<map_t, Tree>::type>>;
            using args_t = meta::transform<meta::force<collapse>::apply, old_args_t>;
            using type = meta::rename<Node, meta::push_front<args_t, fun_t>>;
        };

        template <class Tree>
        struct collapse<Tree> {
            using type = typename meta::if_<has_tmps<Tree>, collapse_impl<lambda, Tree>, meta::lazy::id<Tree>>::type;
        };

        template <class F, class... Trees>
        struct collapse<lambda<F, Trees...>> {
            using type = meta::if_<std::disjunction<is_arg<Trees>...>,
                lambda<F, typename collapse<Trees>::type...>,
                typename collapse_impl<lambda, lambda<F, Trees...>>::type>;
        };

        template <class F, class... Trees>
        struct collapse<tmp<F, Trees...>> {
            using type = meta::if_<std::disjunction<is_arg<Trees>...>,
                tmp<F, typename collapse<Trees>::type...>,
                typename collapse_impl<tmp, lambda<F, Trees...>>::type>;
        };

        template <class T>
        struct expand {
            using type = T;
        };

        template <template <class...> class Node, class... Trees>
        struct expand<Node<Trees...>> {
            using type = Node<typename expand<Trees>::type...>;
        };

        template <auto F, class... Trees>
        struct expand<lambda<meta::val<F>, Trees...>> {
            using type = meta::if_<has_tmps<parse<F, Trees...>>,
                typename expand<normailze_shifts<decltype(F(Trees()...))>>::type,
                lambda<meta::val<F>, typename expand<Trees>::type...>>;
        };

        template <auto F, class... Trees>
        struct expand<tmp<meta::val<F>, Trees...>> {
            using type = meta::if_<has_tmps<parse<F, Trees...>>,
                meta::rename<tmp, typename collapse<typename expand<lambda<meta::val<F>, Trees...>>::type>::type>,
                tmp<meta::val<F>, typename expand<Trees>::type...>>;
        };

        template <class F, class... Trees>
        struct expand<deref<inlined<F, Trees...>>> {
            using type = typename expand<lambda<F, Trees...>>::type;
        };

        template <class Offsets, class F, class... Trees>
        struct expand<deref<shifted<inlined<F, Trees...>, Offsets>>> {
            using type = typename expand<lambda<F, shifted<Trees, Offsets>...>>::type;
        };

        template <class Tree>
        using popup_tmps = typename collapse<typename expand<normailze_shifts<Tree>>::type>::type;

        template <class...>
        struct flatten_nodes;

        template <class V>
        struct flatten_nodes<in<V>> {
            using type = meta::list<>;
        };

        template <template <class...> class Node, class F, class... Trees>
        struct flatten_nodes<Node<F, Trees...>> {
            using type =
                meta::dedup<meta::push_back<meta::concat<typename flatten_nodes<Trees>::type...>, Node<F, Trees...>>>;
        };

        template <class...>
        struct collect_offsets;

        template <class Tmp, class Tree>
        struct collect_offsets<Tmp, Tree> {
            using type = meta::list<>;
        };

        template <class Tmp, template <class...> class Node, class... Trees>
        struct collect_offsets<Tmp, Node<Trees...>> {
            using type = meta::dedup<meta::concat<typename collect_offsets<Tmp, Trees>::type...>>;
        };

        template <class Tmp, auto F, class... Trees>
        struct collect_offsets<Tmp, lambda<meta::val<F>, Trees...>> {
            using type = typename collect_offsets<Tmp, normailze_shifts<decltype(F(Trees()...))>>::type;
        };

        template <class Tmp>
        struct collect_offsets<Tmp, deref<Tmp>> {
            using type = meta::list<meta::val<>>;
        };

        template <class Tmp, class Offsets>
        struct collect_offsets<Tmp, deref<shifted<Tmp, Offsets>>> {
            using type = meta::list<Offsets>;
        };

        template <class Tmp, class F, class... Trees>
        struct collect_offsets<Tmp, deref<inlined<F, Trees...>>> {
            using type = typename collect_offsets<Tmp, lambda<F, Trees...>>::type;
        };

        template <class Tmp, class Offsets, class F, class... Trees>
        struct collect_offsets<Tmp, deref<shifted<inlined<F, Trees...>, Offsets>>> {
            using type = typename collect_offsets<Tmp, lambda<F, shifted<Trees, Offsets>...>>::type;
        };

        template <class T>
        struct dummy_iter {
            friend T fn_deref(dummy_iter) { return {}; }
            friend dummy_iter fn_shift(dummy_iter, ...) { return {}; }
        };

        template <class Map>
        struct make_dummy_iter_f {
            template <class Arg>
            using apply = dummy_iter<meta::third<meta::mp_find<Map, Arg>>>;
        };

        template <class Map>
        struct arg_num_f {
            template <class Arg>
            using apply = meta::second<meta::mp_find<Map, Arg>>;
        };

        template <class Map, class Tree>
        struct make_tmp_record_f {
            template <class Tmp, class Item = meta::mp_find<Map, Tmp>>
            using apply = meta::list<meta::third<Item>,
                typename collect_offsets<Tmp, Tree>::type,
                meta::first<Tmp>,
                meta::list<meta::second<Item>>,
                meta::transform<arg_num_f<Map>::template apply, meta::pop_front<meta::rename<meta::list, Tmp>>>>;
        };

        template <class Map, class Node>
        using ret_type = decltype(std::apply(meta::first<Node>::value,
            meta::transform<make_dummy_iter_f<Map>::template apply,
                meta::pop_front<meta::rename<std::tuple, Node>>>()));

        template <class Map, class Tmp>
        using add_to_arg_map =
            meta::push_back<Map, meta::list<Tmp, typename meta::length<Map>::type, ret_type<Map, Tmp>>>;

        template <class V>
        struct add_f {
            template <class T>
            using apply = std::integral_constant<typename T::value_type, V::value + T::value>;
        };

        template <class Tree,
            class InputTypes,
            class Nodes = typename flatten_nodes<Tree>::type,
            class InitialArgMap = meta::zip<meta::transform<in, meta::make_indices_for<InputTypes>>,
                meta::make_indices_for<InputTypes>,
                InputTypes>,
            class ArgMap = meta::foldl<add_to_arg_map, InitialArgMap, Nodes>>
        using flatten_tmps_tree = meta::transform<make_tmp_record_f<ArgMap, Tree>::template apply, Nodes>;

    } // namespace tmp_impl_

    template <auto F, class... Args>
    constexpr bool has_tmps = tmp_impl_::has_tmps<parse<F, Args...>>::value;

    using tmp_impl_::flatten_tmps_tree;
    using tmp_impl_::popup_tmps;
} // namespace gridtools::fn::ast