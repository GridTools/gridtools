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

#include <memory>
#include <utility>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../dim.hpp"
#include "../sid/allocator.hpp"
#include "../sid/block.hpp"
#include "../sid/concept.hpp"
#include "../sid/contiguous.hpp"
#include "../sid/loop.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "../stage_matrix.hpp"

namespace gridtools {
    namespace x86 {
        struct block_f {
            template <class T>
            auto operator()(T &&data_store) const {
                return sid::block(std::forward<T>(data_store),
                    hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, GT_DEFAULT_TILE_I>,
                        integral_constant<int_t, GT_DEFAULT_TILE_J>>());
            }
        };

        template <class Stages, class Grid, class Allocator>
        auto make_temporaries(Grid const &grid, Allocator &allocator) {
            return tuple_util::transform(
                [&](auto info) {
                    using info_t = decltype(info);
                    using data_t = typename info_t::data_t;
                    using extent_t = typename info_t::extent_t;

                    auto sizes = tuple_util::make<hymap::keys<dim::c, dim::k, dim::j, dim::i, dim::thread>::values>(
                        typename info_t::num_colors_t(),
                        grid.k_total_length(),
                        integral_constant<int_t,
                            GT_DEFAULT_TILE_J + extent_t::jplus::value - extent_t::jminus::value>(),
                        integral_constant<int_t,
                            GT_DEFAULT_TILE_I + extent_t::iplus::value - extent_t::iminus::value>(),
                        omp_get_max_threads());

                    return sid::shift_sid_origin(sid::make_contiguous<data_t, int_t, extent_t>(allocator, sizes),
                        hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, -extent_t::iminus::value>,
                            integral_constant<int_t, -extent_t::jminus::value>>());
                },
                Stages::tmp_plh_map());
        }

        template <class Stage, class Sizes, class ShiftBack>
        struct k_loop_f {
            Sizes m_sizes;
            ShiftBack m_shift_back;

            template <class Ptr, class Strides>
            GT_FORCE_INLINE void operator()(Ptr &ptr, Strides const &strides) const {
                tuple_util::for_each(
                    [&ptr, &strides](auto cell, auto size) {
                        for (int_t k = 0; k < size; ++k) {
                            cell(ptr, strides);
                            cell.inc_k(ptr, strides);
                        }
                    },
                    Stage::cells(),
                    m_sizes);
                sid::shift(ptr, sid::get_stride<dim::k>(strides), m_shift_back);
            }
        };

        template <class Stage, class Sizes, class ShiftBack>
        k_loop_f<Stage, Sizes, ShiftBack> make_k_loop(Sizes sizes, ShiftBack shift_back) {
            return {std::move(sizes), shift_back};
        }

        template <class DataStores>
        struct at_key_f {
            DataStores &m_src;
            template <class Plh>
            GT_FORCE_INLINE decltype(auto) operator()(Plh) const {
                return at_key<Plh>(m_src);
            }
        };

        template <class Stage, class Grid, class DataStores>
        auto make_stage_loop(Stage, Grid const &grid, DataStores &data_stores) {
            using extent_t = typename Stage::extent_t;

            using plhs_t = typename Stage::plhs_t;
            auto composite = tuple_util::convert_to<meta::rename<sid::composite::keys, plhs_t>::template values>(
                tuple_util::transform(at_key_f<DataStores>{data_stores}, plhs_t()));
            using ptr_diff_t = sid::ptr_diff_type<decltype(composite)>;

            auto strides = sid::get_strides(composite);
            ptr_diff_t offset{};
            sid::shift(offset, sid::get_stride<dim::i>(strides), typename extent_t::iminus());
            sid::shift(offset, sid::get_stride<dim::j>(strides), typename extent_t::jminus());
            sid::shift(offset, sid::get_stride<dim::k>(strides), grid.k_start(Stage::interval(), Stage::execution()));

            auto shift_back = -grid.k_size(Stage::interval()) * Stage::k_step();
            auto k_sizes =
                tuple_util::transform([&](auto cell) { return grid.k_size(cell.interval()); }, Stage::cells());

            return [origin = sid::get_origin(composite) + offset,
                       strides = std::move(strides),
                       k_loop = make_k_loop<Stage>(std::move(k_sizes), shift_back)](
                       int_t i_block, int_t j_block, int_t i_size, int_t j_size) {
                ptr_diff_t offset{};
                sid::shift(offset, sid::get_stride<dim::thread>(strides), omp_get_thread_num());
                sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(strides), i_block);
                sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(strides), j_block);
                auto ptr = origin() + offset;

                auto i_loop = sid::make_loop<dim::i>(i_size + extent_t::iplus::value - extent_t::iminus::value);
                auto j_loop = sid::make_loop<dim::j>(j_size + extent_t::jplus::value - extent_t::jminus::value);

                i_loop(j_loop(k_loop))(ptr, strides);
            };
        }

        template <class Spec, class Grid, class DataStores>
        void gridtools_backend_entry_point(backend, Spec, Grid const &grid, DataStores external_data_stores) {
            using stages_t = stage_matrix::make_split_view<Spec>;

            auto alloc = sid::make_cached_allocator(&std::make_unique<char[]>);

            auto data_stores = hymap::concat(tuple_util::transform(block_f(), std::move(external_data_stores)),
                make_temporaries<stages_t>(grid, alloc));

            auto stage_loops = tuple_util::transform(
                [&](auto stage) { return make_stage_loop(stage, grid, data_stores); }, meta::rename<tuple, stages_t>());

            int_t total_i = grid.i_size();
            int_t total_j = grid.j_size();

            int_t NBI = (total_i + GT_DEFAULT_TILE_I - 1) / GT_DEFAULT_TILE_I;
            int_t NBJ = (total_j + GT_DEFAULT_TILE_J - 1) / GT_DEFAULT_TILE_J;

#pragma omp parallel for collapse(2)
            for (int_t bi = 0; bi < NBI; ++bi) {
                for (int_t bj = 0; bj < NBJ; ++bj) {
                    int_t i_size = bi + 1 == NBI ? total_i - bi * GT_DEFAULT_TILE_I : GT_DEFAULT_TILE_I;
                    int_t j_size = bj + 1 == NBJ ? total_j - bj * GT_DEFAULT_TILE_J : GT_DEFAULT_TILE_J;
                    tuple_util::for_each([=](auto &&fun) { fun(bi, bj, i_size, j_size); }, stage_loops);
                }
            }
        }
    } // namespace x86
} // namespace gridtools
