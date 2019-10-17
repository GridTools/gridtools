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

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../common/defs.hpp"
#include "../common/hymap.hpp"
#include "../common/split_args.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "accessor_intent.hpp"
#include "arg.hpp"
#include "compute_extents_metafunctions.hpp"
#include "dim.hpp"
#include "esf_metafunctions.hpp"
#include "extract_placeholders.hpp"
#include "mss.hpp"
#include "positional.hpp"
#include "sid/composite.hpp"

namespace gridtools {
    namespace computation_facade_impl_ {
        template <class Mss>
        using rw_args_from_mss = compute_readwrite_args<typename Mss::esf_sequence_t>;

        template <class Msses,
            class RwArgsLists = meta::transform<rw_args_from_mss, Msses>,
            class RawRwArgs = meta::flatten<RwArgsLists>>
        using all_rw_args = meta::dedup<RawRwArgs>;

        template <class Plh>
        struct ref_generator_f {
            template <class Bound, class Item = meta::mp_find<Bound, Plh>>
            decltype(auto) operator()(Bound &bound) const {
                auto &&res = std::get<meta::st_position<Bound, Item>::value>(bound).m_value;
                return res;
            }
        };

        template <class BoundArgStoragePairs, class MssDescriptors, class EntryPoint, class Grid>
        class computation_facade {
            static_assert(meta::all_of<is_arg_storage_pair, BoundArgStoragePairs>::value, GT_INTERNAL_ERROR);
            static_assert(meta::all_of<is_mss_descriptor, MssDescriptors>::value,
                "make_computation args should be mss descriptors");

            using placeholders_t = extract_placeholders_from_msses<MssDescriptors>;
            using non_tmp_placeholders_t = meta::filter<meta::not_<is_tmp_arg>::apply, placeholders_t>;

            using bound_placeholders_t = meta::transform<meta::first, BoundArgStoragePairs>;

            static_assert(meta::all_of<meta::curry<meta::st_contains, non_tmp_placeholders_t>::template apply,
                              bound_placeholders_t>::value,
                "some bound placeholders are not used in mss descriptors");

            static_assert(meta::is_set_fast<bound_placeholders_t>::value, "bound placeholders should be all different");

            template <class Arg>
            using is_free = negation<meta::st_contains<bound_placeholders_t, Arg>>;

            using free_placeholders_t = meta::filter<is_free, non_tmp_placeholders_t>;

            static_assert(meta::is_empty<free_placeholders_t>(), "");

            using extent_map_t = get_extent_map_from_msses<MssDescriptors>;

            Grid m_grid;
            BoundArgStoragePairs m_bound_data_stores;

            template <class Plh>
            using data_store_ref = std::add_lvalue_reference_t<meta::second<meta::mp_find<BoundArgStoragePairs, Plh>>>;

            using data_store_refs_t = meta::transform<data_store_ref, non_tmp_placeholders_t>;

            using data_store_map_t = hymap::from_keys_values<non_tmp_placeholders_t, data_store_refs_t>;

            data_store_map_t data_store_map() {
                using generators_t = meta::transform<ref_generator_f, non_tmp_placeholders_t>;
                return tuple_util::generate<generators_t, data_store_map_t>(m_bound_data_stores);
            }

          public:
            computation_facade(Grid grid, BoundArgStoragePairs bound_data_stores)
                : m_grid(std::move(grid)), m_bound_data_stores(std::move(bound_data_stores)) {}

            void run() { EntryPoint()(m_grid, data_store_map()); }
        };
    } // namespace computation_facade_impl_

    template <class Backend,
        class EntryPoint,
        class Grid,
        class... Args,
        class BoundArgStoragePairs = meta::filter<is_arg_storage_pair, std::tuple<Args...>>,
        class MssDescriptors = meta::filter<is_mss_descriptor, meta::list<Args...>>>
    computation_facade_impl_::computation_facade<BoundArgStoragePairs, MssDescriptors, EntryPoint, Grid>
    make_computation_facade(Grid grid, Args... args) {
        return {std::move(grid), split_args<is_arg_storage_pair>(std::move(args)...).first};
    }

    template <class... Msses, class Plh>
    constexpr lookup_extent_map<get_extent_map_from_msses<meta::list<Msses...>>, Plh> get_arg_extent(Plh) {
        static_assert(conjunction<is_mss_descriptor<Msses>...>::value,
            "get_arg_extent template arguments should be mss descriptors.");
        static_assert(is_plh<Plh>::value, "get_arg_extent argument should be a placeholder.");
        return {};
    }

    template <class... Msses,
        class Plh,
        class RwPlhs = computation_facade_impl_::all_rw_args<meta::list<Msses...>>,
        intent Intent = meta::st_contains<RwPlhs, Plh>::value ? intent::inout : intent::in>
    constexpr std::integral_constant<intent, Intent> get_arg_intent(Plh) {
        static_assert(conjunction<is_mss_descriptor<Msses>...>::value,
            "get_arg_intent template arguments should be mss descriptors.");
        static_assert(is_plh<Plh>::value, "get_arg_intent argument should be a placeholder.");
        return {};
    }
} // namespace gridtools
