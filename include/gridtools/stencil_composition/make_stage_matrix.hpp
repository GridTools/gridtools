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

#include "../common/defs.hpp"
#include "../meta.hpp"
#include "backend_cuda/need_sync.hpp"
#include "bind_functor_with_interval.hpp"
#include "caches/cache_traits.hpp"
#include "compute_extents_metafunctions.hpp"
#include "dim.hpp"
#include "interval.hpp"
#include "level.hpp"
#include "mss.hpp"
#include "positional.hpp"
#include "stage_matrix.hpp"
#include "structured_grids/stage.hpp"

namespace gridtools {
    namespace make_stage_matrix_impl_ {
        template <class Esf,
            class LevelIndex,
            class Functor = bind_functor_with_interval<typename Esf::esf_function_t, LevelIndex>>
        struct stage_funs {
            using type = meta::list<stage<Functor, typename Esf::args_t>>;
        };

        template <class Esf, class LevelIndex>
        struct stage_funs<Esf, LevelIndex, void> {
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

        template <class Plh, class DataStores, bool = is_tmp_arg<Plh>::value>
        struct get_data_type {
            using sid_t = decltype(at_key<Plh>(std::declval<DataStores>()));
            using type = sid::element_type<sid_t>;
        };

        template <class Plh, class DataStores>
        struct get_data_type<Plh, DataStores, true> {
            using type = typename Plh::data_t;
        };

        template <class Dim>
        using positional_plh_info = stage_matrix::
            plh_info<positional<Dim>, std::false_type, int_t const, extent<>, void, void, integral_constant<uint_t, 1>>;

        template <class EsfExtent, class DataStores, class Mss>
        struct make_plh_info_f {
            using cache_map_t = make_cache_map<typename Mss::cache_sequence_t>;

            template <class Plh,
                class Accessor,
                class IsConst = bool_constant<Accessor::intent_v == intent::in>,
                class Data = typename get_data_type<Plh, DataStores>::type,
                class CacheInfo = meta::mp_find<cache_map_t, Plh, meta::list<void, void, void>>>
            using apply = stage_matrix::plh_info<Plh,
                typename is_tmp_arg<Plh>::type,
                meta::if_<IsConst, std::add_const_t<Data>, Data>,
                sum_extent<EsfExtent, typename Accessor::extent_t>,
                meta::second<CacheInfo>,
                meta::third<CacheInfo>,
                integral_constant<int_t, Plh::location_t::n_colors::value>>;
        };

        template <class Msses, class NeedPositionals, class DataStores, class Mss, class Esf, class NeedSync>
        struct make_cell_f {
            using caches_t = typename Mss::cache_sequence_t;
            using execution_t = typename Mss::execution_engine_t;
            using esf_extent_t = to_horizontal_extent<get_esf_extent<Esf, get_extent_map_from_msses<Msses>>>;

            using esf_plh_map_t = meta::transform<make_plh_info_f<esf_extent_t, DataStores, Mss>::template apply,
                typename Esf::args_t,
                esf_param_list<Esf>>;
            using plh_map_t = meta::if_<NeedPositionals,
                meta::push_back<esf_plh_map_t,
                    positional_plh_info<dim::i>,
                    positional_plh_info<dim::j>,
                    positional_plh_info<dim::k>>,
                esf_plh_map_t>;

            template <class LevelIndex>
            using apply = stage_matrix::cell<typename stage_funs<Esf, LevelIndex>::type,
                interval_from_index<LevelIndex>,
                plh_map_t,
                esf_extent_t,
                typename Mss::execution_engine_t,
                NeedSync>;
        };

        template <class Msses, class NeedPoistionals, class Interval, class DataStores, class Mss>
        struct make_esf_row_f {
            template <class Esf, class NeedSync>
            using apply =
                meta::transform<make_cell_f<Msses, NeedPoistionals, DataStores, Mss, Esf, NeedSync>::template apply,
                    make_level_indices<Interval>>;
        };

        template <class Msses, class NeedPoistionals, class Interval, class DataStores>
        struct make_mss_matrix_f {
            template <class Mss, class Esfs = typename Mss::esf_sequence_t>
            using apply =
                meta::transform<make_esf_row_f<Msses, NeedPoistionals, Interval, DataStores, Mss>::template apply,
                    Esfs,
                    cuda::need_sync<Esfs, typename Mss::cache_sequence_t>>;
        };

        template <class Msses, class NeedPoistionals, class Interval, class DataStores>
        using make_stage_matrices =
            meta::transform<make_mss_matrix_f<Msses, NeedPoistionals, Interval, DataStores>::template apply, Msses>;
    } // namespace make_stage_matrix_impl_
    using make_stage_matrix_impl_::make_stage_matrices;
} // namespace gridtools
