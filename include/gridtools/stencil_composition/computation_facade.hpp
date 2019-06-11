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
#include "../common/timer/timer_traits.hpp"
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

namespace gridtools {

    namespace computation_facade_impl_ {
        template <class Mss>
        using get_esfs = unwrap_independent<typename Mss::esf_sequence_t>;

        template <class Mss>
        using rw_args_from_mss = compute_readwrite_args<get_esfs<Mss>>;

        template <class Msses,
            class RwArgsLists = meta::transform<rw_args_from_mss, Msses>,
            class RawRwArgs = meta::flatten<RwArgsLists>>
        using all_rw_args = meta::dedup<RawRwArgs>;

        template <class Plh>
        struct ref_generator_f {
            template <class Bound,
                class Free,
                class Item = meta::mp_find<Bound, Plh>,
                std::enable_if_t<!std::is_void<Item>::value, int> = 0>
            decltype(auto) operator()(Bound &bound, Free &&) const {
                return (std::get<meta::st_position<Bound, Item>::value>(bound).m_value);
            }
            template <class Bound,
                class Free,
                class Dummy = void,
                class FreeMap = meta::transform<std::decay_t, std::decay_t<Free>>,
                class Item = meta::mp_find<FreeMap, Plh>,
                std::enable_if_t<!std::is_void<Item>::value, int> = 0>
            decltype(auto) operator()(Bound &, Free &&free) const {
                return (std::get<meta::st_position<FreeMap, Item>::value>(std::forward<Free>(free)).m_value);
            }
        };

        template <class BoundArgStoragePairs, class MssDescriptors, class Meter, class Intermediate>
        class computation_facade {

            GT_STATIC_ASSERT((meta::all_of<is_arg_storage_pair, BoundArgStoragePairs>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((meta::all_of<is_mss_descriptor, MssDescriptors>::value),
                "make_computation args should be mss descriptors");

            using placeholders_t = extract_placeholders_from_msses<MssDescriptors>;
            using non_tmp_placeholders_t = meta::filter<meta::not_<is_tmp_arg>::apply, placeholders_t>;

            using bound_placeholders_t = meta::transform<meta::first, BoundArgStoragePairs>;

            GT_STATIC_ASSERT((meta::all_of<meta::curry<meta::st_contains, non_tmp_placeholders_t>::template apply,
                                 bound_placeholders_t>::value),
                "some bound placeholders are not used in mss descriptors");

            GT_STATIC_ASSERT(
                meta::is_set_fast<bound_placeholders_t>::value, "bound placeholders should be all different");

            template <class Arg>
            using is_free = negation<meta::st_contains<bound_placeholders_t, Arg>>;

            using free_placeholders_t = meta::filter<is_free, non_tmp_placeholders_t>;

            using esfs_t = meta::flatten<meta::transform<get_esfs, MssDescriptors>>;

            using extent_map_t = get_extent_map<esfs_t>;

            Meter m_meter;

            Intermediate m_intermediate;
            BoundArgStoragePairs m_bound_data_stores;

            template <class Plh>
            using data_store_ref = std::add_lvalue_reference_t<meta::second<
                meta::mp_find<BoundArgStoragePairs, Plh, meta::list<Plh, typename Plh::data_store_t const>>>>;

            using data_store_refs_t = meta::transform<data_store_ref, non_tmp_placeholders_t>;

            using data_store_map_t = hymap::from_keys_values<non_tmp_placeholders_t, data_store_refs_t>;

            template <class... FreeDataStores>
            data_store_map_t data_store_map(FreeDataStores... srcs) {
                using generators_t = meta::transform<ref_generator_f, non_tmp_placeholders_t>;
                return tuple_util::generate<generators_t, data_store_map_t>(
                    m_bound_data_stores, std::forward_as_tuple(std::move(srcs)...));
            }

          public:
            computation_facade(Intermediate intermediate, BoundArgStoragePairs bound_data_stores)
                : m_meter{"NoName"}, m_intermediate{std::move(intermediate)}, m_bound_data_stores{
                                                                                  std::move(bound_data_stores)} {}

            template <class... Plhs, class... DataStores>
            std::enable_if_t<sizeof...(Plhs) == meta::length<free_placeholders_t>::value> run(
                arg_storage_pair<Plhs, DataStores>... srcs) {
                GT_STATIC_ASSERT((conjunction<meta::st_contains<free_placeholders_t, Plhs>...>::value),
                    "some placeholders are not used in mss descriptors");
                GT_STATIC_ASSERT(
                    meta::is_set_fast<meta::list<Plhs...>>::value, "free placeholders should be all different");
                m_meter.start();
                m_intermediate(data_store_map(std::move(srcs)...));
                m_meter.pause();
            }

            std::string print_meter() const { return m_meter.to_string(); }
            double get_time() const { return m_meter.total_time(); }
            size_t get_count() const { return m_meter.count(); }
            void reset_meter() { m_meter.reset(); }

            template <class Plh,
                class RwPlhs = all_rw_args<MssDescriptors>,
                intent Intent = meta::st_contains<RwPlhs, Plh>::value ? intent::inout : intent::in>
            static constexpr std::integral_constant<intent, Intent> get_arg_intent(Plh) {
                GT_STATIC_ASSERT(is_plh<Plh>::value, "get_arg_intent argument should be a placeholder.");
                return {};
            }

            template <class Plh>
            static constexpr lookup_extent_map<extent_map_t, Plh> get_arg_extent(Plh) {
                GT_STATIC_ASSERT(is_plh<Plh>::value, "get_arg_extent argument should be a placeholder.");
                return {};
            }
        };
    } // namespace computation_facade_impl_

    template <class Backend,
        class Intermediate,
        class... Args,
        class BoundArgStoragePairs = meta::filter<is_arg_storage_pair, std::tuple<Args...>>,
        class MssDescriptors = meta::filter<is_mss_descriptor, meta::list<Args...>>,
        class Meter = typename timer_traits<Backend>::timer_type>
    computation_facade_impl_::computation_facade<BoundArgStoragePairs, MssDescriptors, Meter, Intermediate>
    make_computation_facade(Intermediate intermediate, Args... args) {
        return {std::move(intermediate), split_args<is_arg_storage_pair>(std::move(args)...).first};
    }
} // namespace gridtools
