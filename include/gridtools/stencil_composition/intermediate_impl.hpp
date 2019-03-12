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

#include <boost/fusion/include/at_key.hpp>
#include <boost/fusion/include/count.hpp>
#include <boost/fusion/include/flatten.hpp>
#include <boost/fusion/include/move.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/transform_view.hpp>
#include <boost/optional.hpp>

#include "../common/functional.hpp"
#include "../common/generic_metafunctions/copy_into_set.hpp"
#include "../common/tuple_util.hpp"
#include "../meta/defs.hpp"
#include "./esf_metafunctions.hpp"
#include "./extract_placeholders.hpp"
#include "./local_domain.hpp"
#include "./tmp_storage.hpp"

namespace gridtools {
    namespace _impl {

        /// this functor takes storage infos shared pointers (or types that contain storage_infos);
        /// stashes all infos that are passed through for the first time;
        /// if info is about to pass through twice, the functor substitutes it with the stashed one.
        template <class StorageInfoMap>
        struct dedup_storage_info_f {
            StorageInfoMap &m_storage_info_map;

            template <class Strorage, class StorageInfo>
            data_store<Strorage, StorageInfo> operator()(data_store<Strorage, StorageInfo> const &src) const {
                assert(src.valid());
                static_assert(boost::mpl::has_key<StorageInfoMap, StorageInfo>::value, "");
                auto &stored = boost::fusion::at_key<StorageInfo>(m_storage_info_map);
                if (!stored) {
                    stored = src.get_storage_info_ptr();
                    return src;
                }
                assert(*stored == *src.get_storage_info_ptr());
                return {src, stored};
            }

            template <class Storage, class StorageInfo>
            data_store<Storage, StorageInfo> operator()(data_store<Storage, StorageInfo> &&src) const {
                assert(src.valid());
                static_assert(boost::mpl::has_key<StorageInfoMap, StorageInfo>::value, "");
                auto &stored = boost::fusion::at_key<StorageInfo>(m_storage_info_map);
                if (!stored) {
                    stored = src.get_storage_info_ptr();
                    return src;
                }
                assert(*stored == *src.get_storage_info_ptr());
                return {std::move(src), *stored};
            }

            template <class Arg, class DataStore>
            arg_storage_pair<Arg, DataStore> operator()(arg_storage_pair<Arg, DataStore> const &src) const {
                return this->operator()(src.m_value);
            }
            template <class Arg, class DataStore>
            arg_storage_pair<Arg, DataStore> operator()(arg_storage_pair<Arg, DataStore> &&src) const {
                return this->operator()(std::move(src.m_value));
            }
        };

        template <class Arg>
        struct get_storage_info {
            using type = typename Arg::data_store_t::storage_info_t;
        };

        template <class StorageInfo>
        struct get_storage_info_map_element {
            using type = boost::fusion::pair<StorageInfo, std::shared_ptr<StorageInfo>>;
        };

        template <typename Placeholders>
        using storage_info_map_t = typename boost::fusion::result_of::as_map<boost::mpl::transform_view<
            typename copy_into_set<boost::mpl::transform_view<Placeholders, get_storage_info<boost::mpl::_>>,
                boost::mpl::set0<>>::type,
            get_storage_info_map_element<boost::mpl::_>>>::type;

        // set pointers from the given storage to the local domain
        struct set_arg_store_pair_to_local_domain_f {

            // if the arg belongs to the local domain we set pointers
            template <class Arg, class DataStore, class LocalDomain>
            enable_if_t<meta::st_contains<typename LocalDomain::esf_args, Arg>::value> operator()(
                arg_storage_pair<Arg, DataStore> const &src, LocalDomain &local_domain) const {
                const auto &storage = src.m_value;
                if (storage.device_needs_update())
                    storage.sync();
                auto view = make_target_view(storage);
                namespace f = boost::fusion;
                // here we set data pointers
                f::at_key<Arg>(local_domain.m_local_data_ptrs) = advanced::get_raw_pointer_of(view);
                // here we set meta data pointers
                auto const *storage_info = advanced::storage_info_raw_ptr(view);
                constexpr auto storage_info_index =
                    meta::st_position<typename LocalDomain::storage_info_ptr_list, decltype(storage_info)>::value;
                f::at_c<storage_info_index>(local_domain.m_local_storage_info_ptrs) = storage_info;
                local_domain.m_local_padded_total_lengths[storage_info_index] = storage.info().padded_total_length();
            }
            // do nothing if arg is not in this local domain
            template <class Arg, class DataStore, class LocalDomain>
            enable_if_t<!meta::st_contains<typename LocalDomain::esf_args, Arg>::value> operator()(
                arg_storage_pair<Arg, DataStore> const &, LocalDomain &) const {}
        };

        template <class Srcs, class LocalDomains>
        void update_local_domains(Srcs const &srcs, LocalDomains &local_domains) {
            tuple_util::for_each_in_cartesian_product(set_arg_store_pair_to_local_domain_f{}, srcs, local_domains);
        }

        template <class Mss>
        struct non_cached_tmp_f {
            using local_caches_t = GT_META_CALL(meta::filter, (is_local_cache, typename Mss::cache_sequence_t));
            using cached_args_t = GT_META_CALL(meta::transform, (cache_parameter, local_caches_t));

            template <class Arg>
            GT_META_DEFINE_ALIAS(
                apply, bool_constant, (is_tmp_arg<Arg>::value && !meta::st_contains<cached_args_t, Arg>::value));
        };

        template <class Mss>
        GT_META_DEFINE_ALIAS(extract_non_cached_tmp_args_from_mss,
            meta::filter,
            (non_cached_tmp_f<Mss>::template apply, GT_META_CALL(extract_placeholders_from_mss, Mss)));

        template <class Msses,
            class ArgLists = GT_META_CALL(meta::transform, (extract_non_cached_tmp_args_from_mss, Msses))>
        GT_META_DEFINE_ALIAS(
            extract_non_cached_tmp_args_from_msses, meta::dedup, (GT_META_CALL(meta::flatten, ArgLists)));

        template <class MaxExtent, class Backend>
        struct get_tmp_arg_storage_pair_generator {
            template <class ArgStoragePair>
            struct generator {
                template <class Grid>
                ArgStoragePair operator()(Grid const &grid) const {
                    static constexpr auto backend = Backend{};
                    static constexpr auto arg = typename ArgStoragePair::arg_t{};
                    return make_tmp_data_store<MaxExtent>(backend, arg, grid);
                }
            };

            template <class T>
            GT_META_DEFINE_ALIAS(apply, meta::id, generator<T>);
        };

        template <class MaxExtent, class Backend, class Res, class Grid>
        Res make_tmp_arg_storage_pairs(Grid const &grid) {
            using generators = GT_META_CALL(
                meta::transform, (get_tmp_arg_storage_pair_generator<MaxExtent, Backend>::template apply, Res));
            return tuple_util::generate<generators, Res>(grid);
        }

        template <class MssComponentsList,
            class Extents = GT_META_CALL(
                meta::transform, (get_max_extent_for_tmp_from_mss_components, MssComponentsList))>
        GT_META_DEFINE_ALIAS(get_max_extent_for_tmp, meta::rename, (enclosing_extent, Extents));

        template <class MaxExtent, bool IsStateful>
        struct get_local_domain {
            template <class MssComponents>
            GT_META_DEFINE_ALIAS(apply,
                local_domain,
                (GT_META_CALL(extract_placeholders_from_mss, typename MssComponents::mss_descriptor_t),
                    MaxExtent,
                    IsStateful));
        };

        template <class MssComponentsList,
            bool IsStateful,
            class MaxExtentForTmp = GT_META_CALL(get_max_extent_for_tmp, MssComponentsList),
            class GetLocalDomain = _impl::get_local_domain<MaxExtentForTmp, IsStateful>>
        GT_META_DEFINE_ALIAS(get_local_domains, meta::transform, (GetLocalDomain::template apply, MssComponentsList));

        template <class Mss>
        GT_META_DEFINE_ALIAS(
            rw_args_from_mss, compute_readwrite_args, GT_META_CALL(unwrap_independent, typename Mss::esf_sequence_t));

        template <class Msses,
            class RwArgsLists = GT_META_CALL(meta::transform, (rw_args_from_mss, Msses)),
            class RawRwArgs = GT_META_CALL(meta::flatten, RwArgsLists)>
        GT_META_DEFINE_ALIAS(all_rw_args, meta::dedup, RawRwArgs);

    } // namespace _impl
} // namespace gridtools
