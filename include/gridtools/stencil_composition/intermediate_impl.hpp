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

#include "../common/functional.hpp"
#include "../common/hymap.hpp"
#include "../common/tuple_util.hpp"
#include "../meta/defs.hpp"
#include "../storage/sid.hpp"
#include "esf_metafunctions.hpp"
#include "extract_placeholders.hpp"
#include "local_domain.hpp"
#include "mss_components.hpp"
#include "sid/concept.hpp"
#include "tmp_storage.hpp"

namespace gridtools {
    namespace _impl {
        // set pointers from the given storage to the local domain
        struct set_arg_store_pair_to_local_domain_f {

            // if the arg belongs to the local domain we set pointers
            template <class Arg, class DataStore, class LocalDomain>
            enable_if_t<meta::st_contains<typename LocalDomain::esf_args_t, Arg>::value> operator()(
                arg_storage_pair<Arg, DataStore> const &src, LocalDomain &local_domain) const {
                const auto &storage = src.m_value;
                using strides_kind_t = sid::strides_kind<DataStore>;

                at_key<Arg>(local_domain.m_ptr_holder_map) = sid::get_origin(storage);
                at_key<strides_kind_t>(local_domain.m_strides_map) = sid::get_strides(storage);
                at_key<strides_kind_t>(local_domain.m_total_length_map) = storage.info().padded_total_length();
            }
            // do nothing if arg is not in this local domain
            template <class Arg, class DataStore, class LocalDomain>
            enable_if_t<!meta::st_contains<typename LocalDomain::esf_args_t, Arg>::value> operator()(
                arg_storage_pair<Arg, DataStore> const &, LocalDomain &) const {}
        };

        template <class Srcs, class LocalDomains>
        void update_local_domains(Srcs const &srcs, LocalDomains &local_domains) {
            tuple_util::for_each_in_cartesian_product(set_arg_store_pair_to_local_domain_f{}, srcs, local_domains);
        }

        template <class Mss>
        struct non_cached_tmp_f {
            using local_caches_t = meta::filter<is_local_cache, typename Mss::cache_sequence_t>;
            using cached_args_t = meta::transform<cache_parameter, local_caches_t>;

            template <class Arg>
            GT_META_DEFINE_ALIAS(
                apply, bool_constant, (is_tmp_arg<Arg>::value && !meta::st_contains<cached_args_t, Arg>::value));
        };

        template <class Mss>
        GT_META_DEFINE_ALIAS(extract_non_cached_tmp_args_from_mss,
            meta::filter,
            (non_cached_tmp_f<Mss>::template apply, extract_placeholders_from_mss<Mss>));

        template <class Msses, class ArgLists = meta::transform<extract_non_cached_tmp_args_from_mss, Msses>>
        GT_META_DEFINE_ALIAS(extract_non_cached_tmp_args_from_msses, meta::dedup, (meta::flatten<ArgLists>));

        template <class MaxExtent, class Backend>
        struct get_tmp_arg_storage_pair_generator {
            template <class ArgStoragePair>
            struct generator {
                template <class Grid>
                ArgStoragePair operator()(Grid const &grid) const {
                    return tmp_storage::make_tmp_data_store<MaxExtent>(
                        Backend{}, typename ArgStoragePair::arg_t{}, grid);
                }
            };

            template <class T>
            GT_META_DEFINE_ALIAS(apply, meta::id, generator<T>);
        };

        template <class MaxExtent, class Backend, class Res, class Grid>
        Res make_tmp_arg_storage_pairs(Grid const &grid) {
            using generators =
                meta::transform<get_tmp_arg_storage_pair_generator<MaxExtent, Backend>::template apply, Res>;
            return tuple_util::generate<generators, Res>(grid);
        }

        template <class MssComponentsList,
            class Extents = meta::transform<get_max_extent_for_tmp_from_mss_components, MssComponentsList>>
        GT_META_DEFINE_ALIAS(get_max_extent_for_tmp, meta::rename, (enclosing_extent, Extents));

        template <class Mss>
        GT_META_DEFINE_ALIAS(
            rw_args_from_mss, compute_readwrite_args, unwrap_independent<typename Mss::esf_sequence_t>);

        template <class Msses,
            class RwArgsLists = meta::transform<rw_args_from_mss, Msses>,
            class RawRwArgs = meta::flatten<RwArgsLists>>
        GT_META_DEFINE_ALIAS(all_rw_args, meta::dedup, RawRwArgs);

    } // namespace _impl
} // namespace gridtools
