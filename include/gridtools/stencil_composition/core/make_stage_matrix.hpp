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

#include "../../common/defs.hpp"
#include "../../meta.hpp"
#include "../be_api.hpp"
#include "bind_functor_with_interval.hpp"
#include "cache_info.hpp"
#include "compute_extents_metafunctions.hpp"
#include "interval.hpp"
#include "level.hpp"
#include "mss.hpp"
#include "need_sync.hpp"
#include "stage.hpp"

namespace gridtools {
    namespace core {
        namespace make_stage_matrix_impl_ {
            template <class EsfFunction,
                class Keys,
                class LevelIndex,
                class Functor = bind_functor_with_interval<EsfFunction, LevelIndex>>
            struct stage_funs {
                using type = meta::list<stage<Functor, Keys>>;
            };

            template <class EsfFunction, class Keys, class LevelIndex>
            struct stage_funs<EsfFunction, Keys, LevelIndex, void> {
                using type = meta::list<>;
            };

            template <class From>
            struct make_level_index_f {
                template <class N>
                using apply = level_index<N::value + From::value, From::offset_limit>;
            };

            template <class Interval,
                class From = level_to_index<typename Interval::FromLevel>,
                class To = level_to_index<typename Interval::ToLevel>>
            using make_level_indices = meta::transform<make_level_index_f<From>::template apply,
                meta::make_indices_c<To::value - From::value + 1>>;

            template <class T, class = void>
            struct is_tmp_arg : std::false_type {};

            template <class T>
            struct is_tmp_arg<T, void_t<typename T::tmp_tag>> : std::true_type {};

            template <class Plh, class DataStores, bool = is_tmp_arg<Plh>::value>
            struct get_data_type {
                using sid_t = decltype(at_key<Plh>(std::declval<DataStores>()));
                using type = sid::element_type<sid_t>;
            };

            template <class Plh, class DataStores>
            struct get_data_type<Plh, DataStores, true> {
                using type = typename Plh::data_t;
            };

            template <class Plh, bool = is_tmp_arg<Plh>::value>
            struct get_num_colors {
                using type = void;
            };

            template <class Plh>
            struct get_num_colors<Plh, true> {
                using type = typename Plh::num_colors_t;
            };

            template <class EsfExtent, class DataStores, class Mss>
            struct make_plh_info_f {
                template <class Plh,
                    class Accessor,
                    class CacheInfo = meta::mp_find<typename Mss::cache_map_t, Plh, cache_info<Plh>>>
                using apply = be_api::plh_info<meta::push_front<typename CacheInfo::cache_types_t, Plh>,
                    typename is_tmp_arg<Plh>::type,
                    typename get_data_type<Plh, DataStores>::type,
                    typename get_num_colors<Plh>::type,
                    bool_constant<Accessor::intent_v == intent::in>,
                    sum_extent<EsfExtent, typename Accessor::extent_t>,
                    typename CacheInfo::cache_io_policies_t>;
            };

            template <class Msses, class DataStores, class Mss, class Esf, class NeedSync>
            struct make_cell_f {
                using esf_extent_t = to_horizontal_extent<get_esf_extent<Esf, get_extent_map_from_msses<Msses>>>;

                using plh_map_t = meta::transform<make_plh_info_f<esf_extent_t, DataStores, Mss>::template apply,
                    meta::rename<tuple, typename Esf::args_t>,
                    esf_param_list<Esf>>;

                template <class LevelIndex>
                using apply = be_api::cell<typename stage_funs<typename Esf::esf_function_t,
                                               meta::transform<meta::first, plh_map_t>,
                                               LevelIndex>::type,
                    interval_from_index<LevelIndex>,
                    plh_map_t,
                    esf_extent_t,
                    typename Mss::execution_engine_t,
                    NeedSync>;
            };

            template <class Msses, class Interval, class DataStores, class Mss>
            struct make_esf_row_f {
                template <class Esf, class NeedSync>
                using apply = meta::transform<make_cell_f<Msses, DataStores, Mss, Esf, NeedSync>::template apply,
                    make_level_indices<Interval>>;
            };

            template <class Msses, class Interval, class DataStores>
            struct make_mss_matrix_f {
                template <class Mss, class Esfs = typename Mss::esf_sequence_t>
                using apply = meta::transform<make_esf_row_f<Msses, Interval, DataStores, Mss>::template apply,
                    Esfs,
                    need_sync<Esfs, typename Mss::cache_map_t>>;
            };

            template <class Msses, class Interval, class DataStores>
            using make_stage_matrices =
                meta::transform<make_mss_matrix_f<Msses, Interval, DataStores>::template apply, Msses>;
        } // namespace make_stage_matrix_impl_
        using make_stage_matrix_impl_::make_stage_matrices;
    } // namespace core
} // namespace gridtools
