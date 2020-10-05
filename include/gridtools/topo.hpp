/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#if __cplusplus < 201703
#error requried C++17
#endif

#include <type_traits>

#include "meta.hpp"

namespace gridtools::topo {
    namespace _impl {
        enum class location { vertex, edge, cell };

        template <location...>
        struct entity;

        template <location... Locations>
        struct entity<location::edge, Locations...> {
            using cell = entity<location::cell, location::edge, Locations...>;
            using vertex = entity<location::vertex, location::edge, Locations...>;
        };

        template <location... Locations>
        struct entity<location::cell, Locations...> {
            using edge = entity<location::edge, location::cell, Locations...>;
            using vertex = entity<location::vertex, location::edge, location::cell, Locations...>;
        };

        template <location... Locations>
        struct entity<location::vertex, Locations...> {
            using edge = entity<location::edge, location::vertex, Locations...>;
            using cell = entity<location::cell, location::edge, location::vertex, Locations...>;
        };

        // reverse helper
        template <location... Ts, location... Us>
        entity<Us..., Ts...> operator,(entity<Ts...>, entity<Us...>);

        template <class, location>
        struct push_front;

        template <location T, location... Ts>
        struct push_front<entity<Ts...>, T> {
            using type = entity<T, Ts...>;
        };

        template <class...>
        struct join;

        template <location T, location... Ts, location... Us>
        struct join<entity<T, Ts...>, entity<Us...>> {
            using type = entity<Us..., Ts...>;
        };

        template <location T>
        struct merge_f {
            template <class L>
            using apply = meta::push_front<meta::pop_front<L>, typename push_front<meta::first<L>, T>::type>;
        };

        template <class T>
        struct front_push_f {
            template <class L>
            using apply = meta::push_front<L, T>;
        };

        template <class>
        struct splits;

        template <location T, location U>
        struct splits<entity<T, U>> : meta::list<meta::list<entity<T, U>>> {};

        template <location T, location U, location... Ts>
        struct splits<entity<T, U, Ts...>> {
            using rhs_t = typename splits<entity<U, Ts...>>::type;
            using type = meta::concat<meta::transform<merge_f<T>::template apply, rhs_t>,
                meta::transform<front_push_f<entity<T, U>>::template apply, rhs_t>>;
        };

        template <class>
        struct links;

        template <location T, location U>
        struct links<entity<T, U>> {
            using type = meta::list<entity<T, U>>;
        };

        template <location T, location U, location... Ts>
        struct links<entity<T, U, Ts...>> {
            using type = meta::push_back<typename links<entity<U, Ts...>>::type, entity<T, U>>;
        };

        template <class>
        struct reverse;

        template <location... Locations>
        struct reverse<entity<Locations...>> {
            using type = decltype((entity<Locations>(), ...));
        };

        template <class>
        struct to;

        template <location T, location U, location... Ts>
        struct to<entity<T, U, Ts...>> {
            using type = entity<T>;
        };

        template <class>
        struct is_location : std::false_type {};

        template <location Location>
        struct is_location<entity<Location>> : std::true_type {};

        template <class, class = void>
        struct is_link : std::false_type {};

        template <location U, location T>
        struct is_link<entity<U, T>, std::enable_if_t<U != T && (U == location::edge || T == location::edge)>>
            : std::true_type {};

        template <class T, class = void>
        struct is_chain : is_link<T> {};

        template <location U, location T, location... Ts>
        struct is_chain<entity<U, T, Ts...>, std::enable_if_t<U != T && is_chain<entity<T, Ts...>>::value>>
            : std::true_type {};
    } // namespace _impl

    // topological locations
    using vertex = _impl::entity<_impl::location::vertex>;
    using edge = _impl::entity<_impl::location::edge>;
    using cell = _impl::entity<_impl::location::cell>;

    using locations = meta::list<vertex, edge, cell>;

    //  Neighbor chains are produced with the following syntax:
    //  `vertex::edge:: ... ::vertex::cell`
    //  The meaning is:
    //    the `cell` neighbors of the `vertices` that are neighbors of ... `edges` of the given `vertex`.
    //  The links in the chain from and to the same location type are disallowed.
    //  I.e. `cell::cell` is invalid.
    //
    // Internally chains are canonicalized during creation:
    // `vertex::cell` links of the chain are replaced by equivalent `vertex::edge::cell`.
    // `cell::vertex` links are replaced by `cell::edge::cell`.
    // i.e.  static_assert(std::is_same_v<vertex::cell::vertex, vertex::edge::cell::edge::vertex>);

    // link is the chain with two locations where one of them is `edge`.

    using _impl::is_chain;
    using _impl::is_link;
    using _impl::is_location;

    template <class Chain>
    using reverse = typename _impl::reverse<Chain>::type;

    template <class Chain>
    using to = typename _impl::to<Chain>::type;
    template <class Chain>
    using from = to<reverse<Chain>>;

    // Takes the list of chains and join them together into a one.
    // join<meta::list<vertex::edge, edge::vertex>> => vertex::edge::vertex
    template <class Chains>
    using join = meta::combine<meta::force<_impl::join>::apply, Chains>;

    // Takes a chain and produces the list of links.
    // The result can be joined into the original chain.
    template <class Chain>
    using links = typename _impl::links<Chain>::type;

    // Takes a chain and produces the list of list of chains (splits).
    // Each inner list (split) of the result can be joined into the original chain.
    // All splits are different from each other.
    // Number of splits is exponential against the number of links:
    // meta::length<splits<Chain>>() == 1 << (meta::length<links<Chain>>() - 1)
    template <class Chain>
    using splits = meta::transform<meta::reverse, typename _impl::splits<Chain>::type>;
} // namespace gridtools::topo
