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

#include "../../common/defs.hpp"
#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../dim.hpp"
#include "../pos3.hpp"
#include "../sid/block.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../stage_matrix.hpp"
#include "execinfo_mc.hpp"
#include "loops.hpp"
#include "tmp_storage_sid.hpp"

namespace gridtools {
    namespace mc {
        struct block_f {
            hymap::keys<dim::i, dim::j>::values<int_t, int_t> m_block_sizes;

            block_f(execinfo_mc const &info) : m_block_sizes{info.i_block_size(), info.j_block_size()} {}

            template <class Grid>
            block_f(Grid const &grid) : block_f(execinfo_mc(grid)) {}

            template <class T>
            auto operator()(T &&data_store) const {
                return sid::block(std::forward<T>(data_store), m_block_sizes);
            }
        };

        template <class Stages, class AllParallel, class Grid, class Allocator>
        auto make_temporaries(Grid const &grid, Allocator &allocator) {
            execinfo_mc info(grid);
            return tuple_util::transform(
                [&allocator,
                    block_size = make_pos3(
                        (size_t)info.i_block_size(), (size_t)info.j_block_size(), (size_t)grid.k_size())](auto info) {
                    using info_t = decltype(info);
                    return make_tmp_storage_mc<typename info_t::data_t, typename info_t::extent_t, AllParallel::value>(
                        allocator, block_size);
                },
                Stages::tmp_plh_map());
        }

        template <class DataStores>
        struct at_key_f {
            DataStores &m_src;
            template <class Plh>
            GT_FORCE_INLINE decltype(auto) operator()(Plh) const {
                return at_key<Plh>(m_src);
            }
        };

        template <class Stage, class DataStores>
        auto make_composite(DataStores &data_stores) {
            using plhs_t = typename Stage::plhs_t;
            return tuple_util::convert_to<meta::rename<sid::composite::keys, plhs_t>::template values>(
                tuple_util::transform(at_key_f<DataStores>{data_stores}, plhs_t()));
        }

        template <class Spec, class Grid, class DataStores>
        void gridtools_backend_entry_point(backend, Spec, Grid const &grid, DataStores external_data_stores) {
            using stages_t = stage_matrix::make_split_view<Spec>;
            using all_parrallel_t = typename meta::all_of<execute::is_parallel,
                meta::transform<stage_matrix::get_execution, stages_t>>::type;

            tmp_allocator_mc alloc;

            auto data_stores = hymap::concat(tuple_util::transform(block_f(grid), std::move(external_data_stores)),
                make_temporaries<stages_t, all_parrallel_t>(grid, alloc));

            auto loops = tuple_util::transform(
                [&](auto stage) {
                    using stage_t = decltype(stage);
                    auto k_sizes = tuple_util::transform(
                        [&](auto cell) { return grid.k_size(cell.interval()); }, stage_t::cells());
                    return make_loop<stage_t>(
                        all_parrallel_t(), grid, make_composite<stage_t>(data_stores), std::move(k_sizes));
                },
                meta::rename<tuple, stages_t>());

            run_loops(all_parrallel_t(), grid, std::move(loops));
        }
    } // namespace mc
} // namespace gridtools
