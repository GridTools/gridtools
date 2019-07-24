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

        template <class PlhMap, class Grid, class Allocator>
        auto make_temporaries(Grid const &grid, Allocator &allocator) {
            using plhs_t = meta::transform<stage_matrix::get_plh, PlhMap>;

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
                hymap::from_keys_values<plhs_t, PlhMap>());
        }

        template <class DataStores>
        struct at_key_f {
            DataStores &m_src;
            template <class Plh>
            GT_FORCE_INLINE decltype(auto) operator()(Plh) const {
                return at_key<Plh>(m_src);
            }
        };

        template <class Plhs, class Src>
        auto filter_map(Src &src) {
            return tuple_util::transform(at_key_f<Src>{src}, hymap::from_keys_values<Plhs, Plhs>());
        }

        template <class Size, class Funs>
        struct cell_fun {
            Size m_size;
            template <class Ptr, class Strides>
            GT_FORCE_INLINE void operator()(Ptr const &GT_RESTRICT ptr, Strides const &GT_RESTRICT strides) const {
                for_each<Funs>([&](auto fun) { fun(ptr, strides); });
            }
        };

        template <class Grid>
        struct make_cell_fun_f {
            Grid const &m_grid;
            template <class Cell>
            auto operator()(Cell) const {
                auto size = m_grid.k_size(typename Cell::interval_t());
                return cell_fun<decltype(size), typename Cell::funs_t>{size};
            }
        };

        template <class Stage, class Grid>
        auto make_cell_funs(Grid const &grid) {
            return tuple_util::transform(make_cell_fun_f<Grid>{grid}, meta::rename<tuple, typename Stage::cells_t>());
        }

        template <class KStep, class CellFuns>
        struct k_loop_f {
            CellFuns m_cell_funs;
            int_t m_shift_back;

            template <class Ptr, class Strides>
            GT_FORCE_INLINE void operator()(Ptr &ptr, Strides const &strides) const {
                tuple_util::for_each(
                    [&ptr, &strides](auto fun) {
                        for (int_t k = 0; k < fun.m_size; ++k) {
                            fun(ptr, strides);
                            sid::shift(ptr, sid::get_stride<dim::k>(strides), KStep());
                        }
                    },
                    m_cell_funs);
                sid::shift(ptr, sid::get_stride<dim::k>(strides), m_shift_back);
            }
        };

        template <class KStep, class CellFuns>
        k_loop_f<KStep, CellFuns> make_k_loop(KStep, CellFuns cell_funs, int_t shift_back) {
            return {std::move(cell_funs), shift_back};
        }

        template <class Stage, class Grid, class DataStortes>
        auto make_stage_loop(Stage, Grid const &grid, DataStortes &data_stores) {
            using extent_t = typename Stage::extent_t;
            using execution_t = typename Stage::execution_t;
            using interval_t = typename Stage::interval_t;

            auto composite =
                hymap::concat(sid::composite::keys<>::values<>(), filter_map<typename Stage::plhs_t>(data_stores));
            using ptr_diff_t = sid::ptr_diff_type<decltype(composite)>;

            auto strides = sid::get_strides(composite);
            ptr_diff_t offset{};
            sid::shift(offset, sid::get_stride<dim::i>(strides), typename extent_t::iminus());
            sid::shift(offset, sid::get_stride<dim::j>(strides), typename extent_t::jminus());
            sid::shift(offset, sid::get_stride<dim::k>(strides), grid.k_start(interval_t(), execution_t()));

            auto step = execute::step<execution_t>;
            auto shift_back = -grid.k_size(interval_t()) * step;
            return [origin = sid::get_origin(composite) + offset,
                       strides = std::move(strides),
                       k_loop = make_k_loop(execute::step<execution_t>, make_cell_funs<Stage>(grid), shift_back)](
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
                make_temporaries<typename stages_t::tmp_plh_map_t>(grid, alloc));

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
