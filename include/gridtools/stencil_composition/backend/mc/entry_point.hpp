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

#include <utility>

#include "../../../common/defs.hpp"
#include "../../../common/hymap.hpp"
#include "../../../common/integral_constant.hpp"
#include "../../../common/tuple_util.hpp"
#include "../../../meta.hpp"
#include "../../../sid/as_const.hpp"
#include "../../../sid/block.hpp"
#include "../../../sid/composite.hpp"
#include "../../../sid/concept.hpp"
#include "../../be_api.hpp"
#include "../../common/dim.hpp"
#include "execinfo_mc.hpp"
#include "loops.hpp"
#include "pos3.hpp"
#include "tmp_storage_sid.hpp"

namespace gridtools {
    namespace mc {
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1900
        template <class Grid>
        struct make_size_f {
            Grid const &m_grid;

            template <class Cell>
            auto operator()(Cell) const {
                return m_grid.k_size(Cell::interval());
            }
        };

        template <class DataStores>
        struct make_sid_f {
            DataStores &m_data_stores;
            template <class PtrInfo>
            auto operator()(PtrInfo) const {
                return sid::add_const(PtrInfo::is_const(), at_key<decltype(PtrInfo::plh())>(m_data_stores));
            }
        };
#endif

        struct backend {
            template <class Spec, class Grid, class DataStores>
            friend void gridtools_backend_entry_point(
                backend, Spec, Grid const &grid, DataStores external_data_stores) {
                using stages_t = be_api::make_split_view<Spec>;
                using all_parrallel_t =
                    typename meta::all_of<be_api::is_parallel, meta::transform<be_api::get_execution, stages_t>>::type;

                tmp_allocator_mc alloc;

                execinfo_mc info(grid);

                using tmp_plh_map_t = be_api::remove_caches_from_plh_map<typename stages_t::tmp_plh_map_t>;
                auto temporaries = be_api::make_data_stores(tmp_plh_map_t(),
                    [&alloc,
                        block_size =
                            make_pos3((size_t)info.i_block_size(), (size_t)info.j_block_size(), (size_t)grid.k_size())](
                        auto info) {
                        return make_tmp_storage_mc<decltype(info.data()),
                            decltype(info.extent()),
                            all_parrallel_t::value>(alloc, block_size);
                    });

                auto blocked_externals = tuple_util::transform(
                    [block_size = tuple_util::make<hymap::keys<dim::i, dim::j>::values>(
                         info.i_block_size(), info.j_block_size())](auto &&data_store) {
                        return sid::block(std::forward<decltype(data_store)>(data_store), block_size);
                    },
                    std::move(external_data_stores));

                auto data_stores = hymap::concat(std::move(blocked_externals), std::move(temporaries));

                auto loops = tuple_util::transform(
                    [&](auto stage) {
                        using stage_t = decltype(stage);
                        auto k_sizes = tuple_util::transform(
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1900
                            make_size_f<Grid> { grid }
#else
                            [&](auto cell) { return grid.k_size(cell.interval()); }
#endif
                            ,
                            stage_t::cells());

                        using plh_map_t = typename stage_t::plh_map_t;
                        using keys_t = meta::rename<sid::composite::keys, meta::transform<meta::first, plh_map_t>>;
                        auto composite = tuple_util::convert_to<keys_t::template values>(tuple_util::transform(
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1900
                            make_sid_f<decltype(data_stores)> { data_stores }
#else
                            [&](auto info) {
                                return sid::add_const(info.is_const(), at_key<decltype(info.plh())>(data_stores));
                            }
#endif
                            ,
                            stage_t::plh_map()));
                        return make_loop<stage_t>(all_parrallel_t(), grid, std::move(composite), std::move(k_sizes));
                    },
                    meta::rename<tuple, stages_t>());

                run_loops(all_parrallel_t(), grid, std::move(loops));
            }
        };
    } // namespace mc
} // namespace gridtools
