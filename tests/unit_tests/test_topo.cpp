/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/topo.hpp>

#include <type_traits>

#include <gridtools/meta.hpp>

namespace gridtools::topo {
    namespace {
        // locations
        static_assert(meta::is_set_fast<locations>());
        static_assert(meta::length<locations>() == 3);
        static_assert(meta::st_contains<locations, vertex>());
        static_assert(meta::st_contains<locations, edge>());
        static_assert(meta::st_contains<locations, cell>());

        // is_location
        static_assert(!is_location<int>());
        static_assert(is_location<vertex>());
        static_assert(is_location<edge>());
        static_assert(is_location<cell>());
        static_assert(!is_location<vertex::edge>());

        // is_chain
        static_assert(!is_chain<int>());
        static_assert(!is_chain<vertex>());
        static_assert(is_chain<vertex::edge>());
        static_assert(is_chain<vertex::cell::edge>());

        // is_link
        static_assert(!is_link<int>());
        static_assert(!is_link<vertex>());
        static_assert(is_link<vertex::edge>());
        static_assert(!is_link<vertex::cell::edge>());
        static_assert(!is_link<vertex::cell>());

        // invalid chains. should not compile.
        // TODO(anstaf): Is there the way to check it? run compilation and verify that it fails?
        // using same_location_t = vertex::vertex;

        // chain canonicalization
        static_assert(std::is_same_v<vertex::cell, vertex::edge::cell>);
        static_assert(std::is_same_v<cell::vertex, cell::edge::vertex>);

        // reverse
        static_assert(std::is_same_v<reverse<vertex::edge>, edge::vertex>);
        static_assert(std::is_same_v<reverse<vertex::cell::edge::vertex::cell>, cell::vertex::edge::cell::vertex>);

        // first
        static_assert(std::is_same_v<first<vertex::cell::edge::vertex::cell>, vertex>);

        // last
        static_assert(std::is_same_v<last<vertex::cell::edge::vertex::cell>, cell>);

        // join
        static_assert(std::is_same_v<join<meta::list<vertex::cell, cell::edge, edge::vertex, vertex::cell>>,
            vertex::cell::edge::vertex::cell>);

        // links
        namespace links_test {
            using src_t = vertex::cell::edge::vertex::cell;
            using testee_t = links<src_t>;
            static_assert(meta::length<testee_t>() == 6);
            static_assert(meta::all_of<is_link, testee_t>());
            static_assert(std::is_same_v<join<testee_t>, src_t>);
        } // namespace links_test

        // splits
        namespace splits_test {
            using src_t = vertex::cell::edge::vertex::cell;
            using testee_t = splits<src_t>;
            static_assert(meta::length<testee_t>() == 32);
            static_assert(meta::is_set_fast<testee_t>());

            template <class T>
            using check = std::bool_constant<meta::all_of<is_chain, T>::value && std::is_same_v<join<T>, src_t>>;

            static_assert(meta::all<meta::transform<check, testee_t>>());
        } // namespace splits_test
    }     // namespace
} // namespace gridtools::topo
